from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from tqdm import tqdm
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("model/qwen_tokenizer/")


folder_name = '/data2/wikipedia-cn-20230720-filtered.json'
data = []
offsets = []

current_offset = 0
with open(folder_name, 'r', encoding='utf-8') as file:
    temp=json.load(file)
    for item in tqdm(temp, desc="Processing items"):
        text = item.get('completion', '') + '<|endoftext|>'
        token=tokenizer.encode(text, add_special_tokens=False,max_length=81920,truncation=True)
        if len(token)>5:
            data+=token
            offsets.append((current_offset, len(token)))
            current_offset += len(token)
arr=np.array(data,dtype=np.uint32)
with open('/data3/pretrain_data/wikipedia.bin','wb') as f:
    f.write(arr.tobytes())

with open('/data3/pretrain_data/wikipedia.idx', 'w') as f:
    for offset, length in offsets:
        f.write(f"{offset} {length}\n")