import numpy as np
import glob
from tqdm import tqdm


def merge_bin_files(bin_files, output_bin):
    total_length = 0
    # 首先计算总长度
    for bin_file in bin_files:
        with open(bin_file, 'rb') as f:
            byte_data = f.read()
            total_length += len(byte_data) // 4  # uint32类型每个数字占4个字节
    
    # 创建一个新的.bin文件并写入数据
    with open(output_bin, 'wb') as outfile:
        current_position = 0
        for bin_file in tqdm(bin_files):
            with open(bin_file, 'rb') as infile:
                byte_data = infile.read()
                arr = np.frombuffer(byte_data, dtype=np.uint32)
                outfile.write(arr.tobytes())

def merge_idx_files(idx_files, bin_files, output_idx):
    with open(output_idx, 'w') as out_f:
        current_offset = 0
        for idx_file, bin_file in tqdm(zip(idx_files, bin_files)):
            print(idx_file,bin_file)
            # 计算当前.bin文件中的元素数量
            # 假设每个条目是4字节(uint32)，可以根据实际情况调整
            with open(bin_file, 'rb') as bf:
                byte_data = bf.read()
                num_elements = len(byte_data) // 4  # uint32是4字节
            
            with open(idx_file) as f:
                for line in f:
                    offset, length = map(int, line.split())
                    new_offset = current_offset + offset
                    out_f.write(f"{new_offset} {length}\n")
            
            # 更新current_offset为当前累计的总元素数
            current_offset += num_elements

folder_path = '/data3/pretrain_data/'
# 示例用法：
bin_files = glob.glob(f"{folder_path}/*.bin")
output_bin = '/data3/pretrain_data/pretrain_data.bin'
merge_bin_files(bin_files, output_bin)
# 示例用法：
idx_files=[]
for bin_file in bin_files:
    temp=bin_file[:-3]+'idx'
    idx_files.append(temp)
output_idx = '/data3/pretrain_data/pretrain_data.idx'
merge_idx_files(idx_files, bin_files, output_idx)