#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# 文件路径
file_path = '/instrument/ReRP/dataset/pistachio_zhen/pistachio_2023/pistachio.smi'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"文件不存在: {file_path}")
    exit(1)

# 获取文件大小
file_size = os.path.getsize(file_path)
print(f"文件大小: {file_size / (1024**2):.2f} MB")
print("-" * 80)

# 读取前10行作为示例
print("\n前10行内容:")
print("-" * 80)
with open(file_path, 'r') as f:
    for i, line in enumerate(f, 1):
        if i <= 10:
            print(f"{i}: {line.rstrip()}")
        else:
            break

# 统计总行数
print("\n" + "-" * 80)
print("正在统计总行数...")
line_count = 0
with open(file_path, 'r') as f:
    for line in f:
        line_count += 1

print(f"总行数: {line_count:,}")
print("-" * 80)

# 读取最后10行
print("\n最后10行内容:")
print("-" * 80)
with open(file_path, 'r') as f:
    lines = f.readlines()
    last_10 = lines[-10:] if len(lines) >= 10 else lines
    for i, line in enumerate(last_10, len(lines) - len(last_10) + 1):
        print(f"{i}: {line.rstrip()}")

print("\n" + "=" * 80)
print("读取完成！")

