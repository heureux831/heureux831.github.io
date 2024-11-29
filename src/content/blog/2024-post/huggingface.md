---
title: hugging face 抱抱脸
description: 关于hugging face的工具和使用
date: 2024-11-28
tags:
  - LLM
  - huggingface
image: ./2024.png
authors:
  - Duffy
---

## 什么是hugging face？🤔

## 如何使用？🔧

### hugging face模型下载

> 首先安装相关库
>> `pip install -U huggingface_hub`
>
>
然后编写一个python文件，如下
```python
# 设置环境变量

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download

# 举例
repo_id = "Duffy/Llama-820m"

file_list = [

".gitattributes",

"README.md",

"config.json",

"generation_config.json",

"xxx.py",

"pytorch_model.bin",

"special_tokens_map.json",

"tokenizer.json",

"tokenizer_config.json",

]

local_dir = "./your_model_file_dir"

for file_name in file_list:

hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=local_dir)
```

