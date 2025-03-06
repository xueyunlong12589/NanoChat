import argparse
import random
import time
import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModel
from model.model import NanoChatLM
from model.LMConfig import LMConfig

warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained('./model/qwen_tokenizer')
    # 直接使用您自定义的配置类和模型类
    config = LMConfig.from_pretrained(args.out_dir)
    model = NanoChatLM.from_pretrained(args.out_dir, config=config)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval().to(args.device), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '床前明月光，疑是地上霜。举头望明月，',
            '《小王子》是一本畅销童话书，它讲述了：',
            '马克思主义基本原理是',
            '人类大脑的主要功能是',
            '万有引力原理是',
            '世界上最高的山峰是',
            '地球上最大的动物是',
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                "你的创作者是谁？你是什么人工智能助手？",
                "用一句话描述“拥有梦想”的含义。",
                "“百货商场”这个名词来源于哪？",
                "给出以下分数的平均数: 85, 90, 92, 88, 95。\n",
                "给我介绍一个中国古代历史上的女性英雄。",
                "请写一段python快速排序的代码",
                "说出一个19世纪的美国发明。",
                "你是ChatGPT吧。",
                '孕妇在饮食上需要注意什么？',
                '老年人如何预防骨质疏松？',
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
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


def main():
    parser = argparse.ArgumentParser(description="Chat with MiniMind")
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out_sft/', type=str)
    parser.add_argument("--seed", type=int, default=1337)#32
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--dim', default=896, type=int)
    parser.add_argument('--n_layers', default=24, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--stream', default=True, type=bool)
    parser.add_argument('--model_mode', default=1, type=int,
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型")
    args = parser.parse_args()
    set_seed(args.seed)

    model, tokenizer = init_model(args)

    prompts = get_prompt_datas(args)
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        if test_mode == 0: print(f'👶: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})
        messages=[{'role':'system','content':'你是NanoChat，由xueyunlong创造。你是一个乐于助人的人工智能助手。'}]+messages

        new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-args.max_seq_len + 1:] if args.model_mode != 0 else (prompt)

        new_prompt=new_prompt[:-1]

        answer = new_prompt
        with torch.no_grad():
            x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_seq_len,
                temperature=args.temperature,
                top_p=args.top_p,
                stream=True,
                # rp=1.5,
                # do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

            print('🤖️: ', end='')
            try:
                if not args.stream:
                    print(tokenizer.decode(outputs.squeeze()[x.shape[1]:].tolist(), skip_special_tokens=True), end='')
                else:
                    history_idx = 0
                    for y in outputs:
                        answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                        if (answer and answer[-1] == '�') or not answer:
                            continue
                        print(answer[history_idx:], end='', flush=True)
                        history_idx = len(answer)
            except StopIteration:
                print("No answer")
            print('\n')

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
