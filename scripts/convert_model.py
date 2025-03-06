import torch
import warnings
import sys
import os

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.LMConfig import LMConfig
from model.model import NanoChatLM

warnings.filterwarnings('ignore', category=UserWarning)


def convert_huggingface(model_path, transformers_path):
    def export_tokenizer(transformers_path):
        tokenizer = AutoTokenizer.from_pretrained('../model/qwen_tokenizer')
        tokenizer.save_pretrained(transformers_path)

    LMConfig.register_for_auto_class()
    NanoChatLM.register_for_auto_class("AutoModelForCausalLM")

    config = LMConfig.from_pretrained(model_path)
    lm_model = NanoChatLM.from_pretrained(model_path, config=config)

    
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    export_tokenizer(transformers_path)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")





if __name__ == '__main__':
    lm_config = LMConfig(dim=896, n_layers=24, max_seq_len=512, use_moe=False)

    model_path = '../out'

    transformers_path = '../NanoChat-0.3B-base'

    convert_huggingface(model_path, transformers_path)


