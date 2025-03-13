import os
import platform
import argparse
import time
import math
import random
import numpy as np
import torch.nn.functional as F
import warnings
import pandas as pd
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from model.model import NanoChatLM
from model.LMConfig import LMConfig
from model.dataset import DPODataset
from accelerate import Accelerator

warnings.filterwarnings('ignore')
# os.environ['CUDA_LAUNCH_BLOCKING']='1'



def get_cosine_with_min_lr_schedule_with_warmup(optimizer,num_training_steps,warmap_ratio=0.01,cosine_ratio=0.9,min_lr_ratio=0.1):
    num_warmup_steps = int(num_training_steps * warmap_ratio)
    num_cosine_step=int(num_training_steps*cosine_ratio)
    
    def inner(current_step):
        if current_step < num_warmup_steps:
            # 在预热阶段线性增加学习率
            res=current_step / max(1,num_warmup_steps)
        # 计算总的学习率比例，采用余弦退火策略
        elif current_step < num_cosine_step+num_warmup_steps:
            progress = (current_step - num_warmup_steps) / (num_cosine_step)
            cosine_decay = 0.5 * (1. + math.cos(math.pi * progress))
            res = min_lr_ratio+(1-min_lr_ratio)*cosine_decay
        else:
            res=min_lr_ratio
        return res
    return lr_scheduler.LambdaLR(optimizer,inner)

def logits_to_probs(logits, labels):
    # logits shape: (batch_size, seq_len, vocab_size)
    # labels shape: (batch_size, seq_len)
    # probs shape: (batch_size, seq_len)
    log_probs = F.log_softmax(logits, dim=2)
    probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return probs


def dpo_loss(ref_probs, probs, beta):
    # ref_probs 和 probs 都是 shape: (batch_size, seq_len)
    # 计算每个样本的平均概率
    ref_probs = ref_probs.mean(dim=1)
    probs = probs.mean(dim=1)

    # 将 chosen 和 rejected 数据分开
    batch_size = ref_probs.shape[0]
    chosen_ref_probs = ref_probs[:batch_size // 2]
    reject_ref_probs = ref_probs[batch_size // 2:]
    chosen_probs = probs[:batch_size // 2]
    reject_probs = probs[batch_size // 2:]

    pi_logratios = chosen_probs - reject_probs
    ref_logratios = chosen_ref_probs - reject_ref_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch):
    if args.resume_from_checkpoint and epoch==resume_epoch and resume_step!=0:
        activated_dataloader=accelerator.skip_first_batches(train_loader,resume_step * accelerator.gradient_accumulation_steps)
        current_step = resume_step
    else:
        activated_dataloader=train_loader
        current_step = 0    
    start_time = time.time()
    for iter, batch in enumerate(activated_dataloader):
        x_chosen = batch['x_chosen']
        x_rejected = batch['x_rejected']
        y_chosen = batch['y_chosen']
        y_rejected = batch['y_rejected']
        mask_chosen = batch['mask_chosen']
        mask_rejected = batch['mask_rejected']
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)
        with accelerator.accumulate(model):

            with accelerator.autocast():
                with torch.no_grad():
                    ref_outputs = ref_model(x)
                    ref_logits = ref_outputs.logits
                ref_probs = logits_to_probs(ref_logits, y)
                ref_probs = ref_probs * mask
                outputs = model(x)
                logits = outputs.logits
                probs = logits_to_probs(logits, y)
                probs = probs * mask
                loss = dpo_loss(ref_probs, probs, beta=0.1)
                loss = loss / args.accumulation_steps

        accelerator.backward(loss)
        

        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), args.grad_clip)
        
        optimizer.step()
        if (epoch*len(train_loader)+iter+1) % args.accumulation_steps==0:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            current_step += 1
            global_step = current_step + epoch * steps_per_epoch
            if (global_step) % args.log_interval == 0:
                spend_time = time.time() - start_time
                mean_loss = accelerator.reduce(loss, "mean")
                accelerator.print(
                    'Epoch:[{}/{}]({}/{}) global_step:{} loss:{:.5f} lr:{:.12f} epoch_Time:{}min:'.format(
                        epoch + 1,
                        args.epochs,
                        current_step,
                        steps_per_epoch,
                        global_step,
                        mean_loss.item(),
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (current_step) * steps_per_epoch // 60 - spend_time // 60))

                if args.use_wandb:
                    accelerator.log({"loss": mean_loss.item(),
                            "lr": optimizer.param_groups[-1]['lr'],
                            "epoch_Time": spend_time / (current_step) * steps_per_epoch // 60 - spend_time // 60})

            if (global_step) % args.save_interval == 0:
                    accelerator.print(f"save model -> step_{global_step}")
                    accelerator.unwrap_model(model).save_pretrained(
                        save_directory=args.out_dir,
                        is_main_process=accelerator.is_main_process,
                        state_dict=accelerator.get_state_dict(model),
                        save_function=accelerator.save,
                        safe_serialization=False
                    )
                
                # if (global_step) % checkpoint_step == 0:
                #     accelerator.print(f"save state_dict -> step_{global_step}")
                #     accelerator.save_state(args.out_dir+f"step_{global_step}")
    if epoch==(args.epochs-1):
        accelerator.print("save model -> final")
        accelerator.unwrap_model(model).save_pretrained(
            save_directory=args.out_dir,
            is_main_process=accelerator.is_main_process,
            state_dict=accelerator.get_state_dict(model),
            save_function=accelerator.save,
            safe_serialization=False
        )


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained('./model/qwen_tokenizer')
    config = LMConfig.from_pretrained('out_sft/')
    model = NanoChatLM.from_pretrained('out_sft/', config=config)
    model.train()
    model.requires_grad_(True)
    
    # 初始化参考模型
    ref_config = LMConfig.from_pretrained('out_sft/')
    ref_model = NanoChatLM.from_pretrained('out_sft/', config=ref_config)
    
    ref_model.eval()
    ref_model.requires_grad_(False)

    accelerator.print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind RLHF")
    parser.add_argument("--out_dir", type=str, default="out_rlhf_4")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)#32
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--wandb_project", type=str, default="MiniMind-RLHF")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--accumulation_steps", type=int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--checkpoint_nums", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument('--dim', default=896, type=int)
    parser.add_argument('--n_layers', default=24, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--attention', default='GQA', type=str)#MLA GQA
    parser.add_argument("--data_path", type=str, default="./dataset/ultrafeedback_zh_binarized_lowest.json")

    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accumulation_steps,
        mixed_precision=args.dtype,
        log_with="wandb" if args.use_wandb else None
        )
    args.device = accelerator.device
    lm_config = LMConfig(dim=args.dim, n_layers=args.n_layers, max_seq_len=args.max_seq_len, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * lm_config.max_seq_len*args.accumulation_steps
    set_seed(args.seed)

    accelerator.init_trackers(project_name=args.wandb_project)

    model, ref_model, tokenizer = init_model(lm_config)

    train_ds = DPODataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=0.1)

    model, optimizer, train_loader= accelerator.prepare(model, optimizer, train_loader)

    steps_per_epoch = len(train_loader)//args.accumulation_steps
    checkpoint_step=(steps_per_epoch*args.epochs)//args.checkpoint_nums
    scheduler= get_cosine_with_min_lr_schedule_with_warmup(
        optimizer,
        num_training_steps=steps_per_epoch* args.epochs,
    )  
    # scheduler=accelerator.prepare(scheduler)

    resume_epoch=0
    resume_step=0
    if args.resume_from_checkpoint is not None:
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(args.resume_from_checkpoint.split("step_")[-1])
        resume_epoch = global_step // steps_per_epoch
        resume_step = int(global_step % steps_per_epoch)
        accelerator.print(f"resume from checkpoint -> {args.resume_from_checkpoint}")

    for epoch in range(resume_epoch,args.epochs):
        train_epoch(epoch)
    accelerator.end_training()

