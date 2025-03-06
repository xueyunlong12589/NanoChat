import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MyPretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path+'.idx', 'r') as f:
            self.offsets = [tuple(map(int, line.strip().split())) for line in f.readlines()]
        self.data = np.memmap(data_path+'.bin', dtype=np.dtype('uint32'), mode='r')
        print(len(self.offsets))

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        start, length = self.offsets[index]

        token_ids=self.data[start:start + length].tolist()
        if len(token_ids) > self.max_length+1:
            token_ids = token_ids[:self.max_length+1]
        # 填充较短的序列
        padding_length = self.max_length - len(token_ids)+1
        if padding_length > 0:
            token_ids += [self.tokenizer.pad_token_id] * padding_length
        loss_mask = np.array(token_ids) != self.tokenizer.pad_token_id

        X = torch.tensor(token_ids[:-1], dtype=torch.long)
        Y = torch.tensor(token_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, messages):
        """构建符合ChatML格式的对话"""
        messages=[{'role':'system','content':'你是NanoChat，由xueyunlong创造。你是一个乐于助人的人工智能助手。'}]+messages
        input=self.tokenizer.apply_chat_template(messages,add_generation_prompt=False,tokenize=False)
        input=input[:-1]
        input_ids=self.tokenizer.encode(input)
        return input_ids

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        input_ids = self._create_chat_prompt(sample['conversations'])
        input_ids = input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = np.array(input_ids) != self.tokenizer.pad_token_id

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask


class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        prompt = []
        for temp in item["conversations"]:
            if temp['from'] == 'system':
                prompt.append({'role': 'system', 'content': temp['value']})
            else:
                prompt.append({'role': 'user', 'content': temp['value']})
        chosen=prompt+[{'role': 'assistant', 'content': item["chosen"]["value"]}]
        rejected=prompt+[{'role': 'assistant', 'content': item["rejected"]["value"]}]
        if chosen[0]['role']!='system':
            chosen=[{'role':'system','content':'你是NanoChat，由xueyunlong创造。你是一个乐于助人的人工智能助手。'}]+chosen
            rejected=[{'role':'system','content':'你是NanoChat，由xueyunlong创造。你是一个乐于助人的人工智能助手。'}]+rejected

        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = chosen_prompt[:-1]
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = rejected_prompt[:-1]
        chosen_encoding = self.tokenizer.encode(chosen_prompt)
        rejected_encoding = self.tokenizer.encode(rejected_prompt)

        chosen_input_ids = chosen_encoding[:self.max_length]
        rejected_input_ids = rejected_encoding[:self.max_length]
        chosen_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(chosen_input_ids))
        rejected_input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(rejected_input_ids))
        chosen_loss_mask = np.array(chosen_input_ids) != self.tokenizer.pad_token_id
        rejected_loss_mask = np.array(rejected_input_ids) != self.tokenizer.pad_token_id
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }


if __name__ == "__main__":
    pass
