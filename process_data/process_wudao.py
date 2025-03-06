from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import numpy as np
import gc
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("model/qwen_tokenizer/")


folder_path = '/data2/WuDaoCorpus2.0_base_200G/'

def process_files(folder_path_list,i):
    data = []
    offsets = []
    current_offset = 0
    for j,filename in enumerate(folder_path_list):
        print(i,j)
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                temp=json.load(file)
                print(len(temp))
                for k,item in enumerate(temp):
                    # print(i,j,k)
                    text = item.get('content', '') + '<|endoftext|>'
                    token=tokenizer.encode(text, add_special_tokens=False,max_length=81920,truncation=True)
                    if len(token)>5:
                        data+=token
                        offsets.append((current_offset, len(token)))
                        current_offset += len(token)
    arr=np.array(data,dtype=np.uint32)
    with open('/data3/pretrain_data/wudao_'+str(i)+'.bin','wb') as f:
        f.write(arr.tobytes())

    with open('/data3/pretrain_data/wudao_'+str(i)+'.idx', 'w') as f:
        for offset, length in offsets:
            f.write(f"{offset} {length}\n")

folder_list=os.listdir(folder_path)
folder_list=folder_list[3*37:]
print(len(folder_list))

n_folder=15
size=int(np.ceil(len(folder_list)/n_folder))
print(size)
for i in range(n_folder):
    if i==size:
        temp=folder_list[i*size:]
    else:
        temp=folder_list[i*size:(i+1)*size]
    process_files(temp,i+4)

