import numpy as np

file_path = "/fs_mol/linhaitao/synflow_mix/data/all/filtered_train_reactions_0.8.txt"

# 首先统计总行数
with open(file_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

# 生成随机索引
index_permute = np.random.permutation(total_lines)
val_indices = set(index_permute[:2000].tolist())
train_indices = set(index_permute[2000:].tolist())

# 一次遍历文件，根据索引直接写入对应文件
with open(file_path, "r", encoding="utf-8") as f_in, \
     open('/fs_mol/linhaitao/synflow_mix/data/all/train.txt', 'w', encoding='utf-8') as f_train, \
     open('/fs_mol/linhaitao/synflow_mix/data/all/val.txt', 'w', encoding='utf-8') as f_val:
    for idx, line in enumerate(f_in):
        line = line.strip()
        if idx in val_indices:
            f_val.write(line + "\n")
        elif idx in train_indices:
            f_train.write(line + "\n")


test_file_path = '/fs_mol/linhaitao/synflow_mix/data/all/all_test_reactions_0.8.txt'
with open(test_file_path, "r", encoding="utf-8") as f_in, \
     open('/fs_mol/linhaitao/synflow_mix/data/all/test.txt', 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        f_out.write(line + "\n")
