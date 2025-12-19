import json
import os

data_file = 'weibo_data_20251218_012526.json'

# 检查文件大小
size = os.path.getsize(data_file)
print(f'文件大小: {size / 1024 / 1024:.2f} MB')

# 加载数据
with open(data_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'\nJSON文件中的记录数: {len(data)}')

# 统计各关键词数量
keywords = {}
for d in data:
    kw = d.get('keyword', 'unknown')
    keywords[kw] = keywords.get(kw, 0) + 1

print(f'\n各关键词数量（共{len(keywords)}个关键词）:')
for k, v in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
    print(f'  {k}: {v} 条')

# 检查是否有重复ID
ids = [d.get('id') for d in data]
unique_ids = set(ids)
print(f'\n唯一ID数量: {len(unique_ids)}')
print(f'重复ID数量: {len(ids) - len(unique_ids)}')

# 检查数据格式
if len(data) > 0:
    print(f'\n第一条数据示例:')
    print(json.dumps(data[0], ensure_ascii=False, indent=2))

