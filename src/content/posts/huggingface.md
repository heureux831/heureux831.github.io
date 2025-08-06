---
title: hugging face 抱抱脸
description: 关于hugging face的工具和使用
published: 2024-11-28
tags:
  - LLM
  - huggingface
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


## Hugging Face Hub

Hugging Face Hub 是一个集成模型、数据集、和其他机器学习资源的平台，它使得分享和使用模型变得非常方便。Hub 提供了易于使用的 API、GitHub 风格的模型管理，并且是目前许多 NLP 和计算机视觉任务中最流行的模型资源。

可以通过访问 [Hugging Face Hub](https://huggingface.co/models) 页面浏览大量的预训练模型。这些模型可以按任务类别、框架（例如 TensorFlow、PyTorch）和领域（如 NLP、计算机视觉）进行筛选。

每个模型都有详细的文档README file，描述模型的用途、性能和限制。

Hugging Face 提供了 transformers 库，通过简单的几行代码，你可以轻松加载和使用预训练模型。

使用 AutoModelForXxx 和 AutoTokenizer 来加载适合你任务的模型和对应的 tokenizer。

如果我希望加载一个文本分类模型（如 BERT）：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

这两行代码会自动从 Hugging Face Hub 下载 `bert-base-uncased `模型及其对应的 tokenizer，并准备好进行后续推理或微调。
如果存在网络问题，我们可以下载模型到本地，然后把模型id换为模型地址。

##### 上传自己的模型

如果你训练了自己的模型，并希望将其上传到 Hugging Face Hub，与他人共享或用于在线推理，可以使用 Hugging Face 的 CLI 工具。
  
```bash
#  先安装 Hugging Face CLI 工具：
pip install huggingface_hub
# 然后登录 Hugging Face：
huggingface-cli login
# 需要提供 Hugging Face 账户的 token，登录后即可上传模型。
# 上传模型的方法：
huggingface-cli upload ./path_to_model
```
  

这样，你可以把自己训练的模型上传到 Hugging Face Hub，其他用户也能下载并使用。

##### 创建和管理模型库

• Hugging Face Hub 不仅仅是一个模型仓库，它也支持模型版本管理，你可以对上传的模型进行版本管理和更新。

• 每次上传新版本的模型时，Hugging Face 会为你自动保存版本，允许你方便地回退到之前的版本。

  
#### Transformers 库

transformers 是 Hugging Face 提供的一个重要库，它涵盖了很多自然语言处理（NLP）模型的实现，并提供了简洁的 API 来加载、微调和使用这些模型。这个库支持多种任务，如文本生成、分类、情感分析、翻译、命名实体识别等。

  

##### 加载预训练模型

transformers 库的核心特性之一是通过 AutoModel 和 AutoTokenizer 类，轻松加载预训练模型和对应的 tokenizer。AutoModel 是一个通用类，它会根据你指定的模型名称自动加载合适的模型。

**使用方法**：

• 假设你想加载一个用于文本分类的 BERT 模型：

  

from transformers import AutoModelForSequenceClassification, AutoTokenizer

  

model_name = "bert-base-uncased"  _# 这里是模型的名称_

model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name)

  

• AutoModelForSequenceClassification 会根据模型名称下载适合文本分类任务的模型结构。

• AutoTokenizer 会加载适配模型的 tokenizer（分词器），并准备好对输入文本进行编码。

  

**2.2 进行推理（Inference）**

  

推理过程是指将一个输入传入模型，得到模型的输出。transformers 库提供了非常简便的接口来进行推理。

  

**步骤**：

• 假设你想使用 BERT 进行情感分析：

  
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import torch

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  

# 编码输入文本

inputs = tokenizer("I love Hugging Face!", return_tensors="pt")

# 推理
with torch.no_grad():

    logits = model(**inputs).logits

  

_# 获取预测结果_

predicted_class = logits.argmax().item()

print(f"Predicted class: {predicted_class}")
```
  

• 在这段代码中，我们首先通过 tokenizer 编码了输入文本 I love Hugging Face!，然后将其传递给模型进行推理，最后从模型输出中获取预测的类标签。

  

**2.3 Fine-tuning（微调）**

  

微调是指将预训练模型根据自己的任务和数据进行进一步训练。在 transformers 中，微调过程可以通过 Trainer API 简化。Trainer 提供了很多功能，包括训练过程管理、评估、日志记录等。

  

**步骤**：

• 假设你有一个分类任务数据集，你可以使用 Trainer 来微调 BERT 模型：

  

from transformers import Trainer, TrainingArguments

  

_# 设置训练参数_

training_args = TrainingArguments(

    output_dir="./results",          _# 模型输出文件夹_

    evaluation_strategy="epoch",     _# 每个 epoch 后评估_

    per_device_train_batch_size=8,   _# 每个设备的训练批次大小_

    per_device_eval_batch_size=8,    _# 每个设备的评估批次大小_

    logging_dir="./logs",            _# 日志文件夹_

)

  

trainer = Trainer(

    model=model,                       _# 你的模型_

    args=training_args,                _# 训练参数_

    train_dataset=train_dataset,       _# 训练数据集_

    eval_dataset=eval_dataset,         _# 验证数据集_

)

  

trainer.train()

  

• 在这个代码中，Trainer 会管理整个训练过程，包括从数据加载到模型训练，并在每个 epoch 后进行评估。你只需要指定训练参数和数据集。

  
好的！我们继续深入介绍 Hugging Face 的其他主要功能，包括 **Datasets 库**、**Tokenizers 库**、**Inference API** 等。

  

**3. Datasets 库**

  

Hugging Face 的 datasets 库提供了一个简洁的接口来访问和处理各种公共数据集，帮助开发者快速加载和使用数据集进行训练和评估。这个库包括了许多常用的 NLP 数据集，比如 GLUE、SQuAD、IMDB 等，也支持用户上传自己的数据集。

  

**3.1 加载公共数据集**

  

你可以通过 load_dataset 函数直接加载 Hugging Face Hub 上的各种数据集。这个库支持很多格式的数据集，比如 CSV、JSON、文本文件等，也支持根据任务进行预处理。

  

**使用方法**：

• 假设你要加载 IMDB 数据集：

  

from datasets import load_dataset

  

dataset = load_dataset("imdb")

print(dataset)

  

这段代码会自动从 Hugging Face Hub 下载并加载 IMDB 数据集。load_dataset 会返回一个字典对象，包含了训练集、验证集等不同的分区。

  

• 你可以查看数据集的内容：

  

print(dataset['train'][0])  _# 查看训练集的第一个样本_

  

  

  

**3.2 加载自定义数据集**

  

如果你有自己的数据集，比如 CSV 文件，可以通过 Dataset 类将其转换为 Hugging Face 格式，以便于处理和使用。

  

**使用方法**：

• 假设你有一个包含文本和标签的 CSV 文件，你可以将其加载为 Hugging Face 数据集：

  

from datasets import Dataset

import pandas as pd

  

_# 加载 CSV 文件为 pandas DataFrame_

df = pd.read_csv("your_dataset.csv")

  

_# 将 DataFrame 转换为 Hugging Face Dataset 格式_

dataset = Dataset.from_pandas(df)

print(dataset)

  

  

• 如果你的数据集是存储在 JSON 文件中，也可以通过类似的方法加载：

  

dataset = load_dataset("json", data_files="your_dataset.json")

  

  

  

**3.3 数据集操作**

  

Hugging Face 的 datasets 库支持对数据集进行各种操作，如筛选、拆分、批处理等。

• **数据集拆分**：

如果你希望将数据集划分为训练集、验证集和测试集，可以使用 train_test_split 方法：

  

dataset = load_dataset("imdb")

train_test = dataset["train"].train_test_split(test_size=0.2)

print(train_test)

  

  

• **过滤数据**：

如果你想基于特定条件筛选数据集中的样本：

  

dataset_filtered = dataset.filter(lambda example: example['label'] == 1)

print(dataset_filtered)

  

  

• **映射函数**：

你可以对数据集进行预处理操作，比如文本清洗或分词：

  

def preprocess_function(examples):

    return tokenizer(examples["text"], padding="max_length", truncation=True)

  

dataset = dataset.map(preprocess_function, batched=True)

  

  

  

**3.4 保存和共享数据集**

  

你可以将处理好的数据集保存为本地文件，并上传到 Hugging Face Hub 进行分享。

  

**保存为 CSV 文件**：

  

dataset.to_csv("processed_dataset.csv")

  

**上传到 Hugging Face Hub**：

• 在 Hugging Face Hub 上创建一个新的数据集页面并上传自己的数据集。

  

**4. Tokenizers 库**

  

Hugging Face 的 tokenizers 库是一个高效的工具，用于处理文本的分词和编码。它支持各种分词方法，如 BPE（Byte Pair Encoding）、WordPiece、SentencePiece 等，可以帮助你高效地处理文本数据。

  

**4.1 加载和使用 Tokenizer**

  

Hugging Face 提供了多种预训练的 tokenizer，你可以使用 AutoTokenizer 类来加载适配不同模型的分词器。

  

**使用方法**：

• 假设你想使用 BERT 模型的 tokenizer：

  

from transformers import AutoTokenizer

  

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

encoding = tokenizer("Hello, Hugging Face!")

print(encoding)

  

这会将输入文本 "Hello, Hugging Face!" 转换为模型能够理解的 token 格式，并输出一个包含 token IDs 的字典。

  

**4.2 自定义 Tokenizer**

  

如果你想自定义一个 tokenizer，可以使用 tokenizers 库提供的低级 API。你可以选择不同的分词算法（如 BPE、WordPiece 等），并根据自己的语料库训练一个新的 tokenizer。

  

**创建自定义 BPE Tokenizer**：

  

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

  

_# 创建一个空的 BPE 模型_

tokenizer = Tokenizer(models.BPE())

  

_# 设置分词器的预处理步骤_

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

  

_# 创建一个训练器，设置特殊 token_

trainer = trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]"])

  

_# 训练 tokenizer_

tokenizer.train_from_file("your_data.txt", trainer=trainer)

  

_# 保存模型_

tokenizer.save("custom_tokenizer.json")

  

• 这段代码会使用你的数据（your_data.txt）训练一个 BPE 分词器，并保存为 custom_tokenizer.json 文件。

  

**4.3 Tokenizer 的高级功能**

• **分词**：

  

tokens = tokenizer.encode("Hello, Hugging Face!")

print(tokens.tokens)  _# 输出 token 列表_

  

  

• **解码**：

  

decoded_text = tokenizer.decode(tokens.ids)

print(decoded_text)  _# 输出原始文本_

  

  

• **编码和解码批量数据**：

你可以对多个文本样本进行批量编码和解码：

  

texts = ["Hello, Hugging Face!", "I love transformers!"]

encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

print(encodings)

  

**5. Inference API (推理 API)**

  

Hugging Face 提供了一个强大的推理 API，用户可以通过简单的 HTTP 请求将数据传给在线模型，获取推理结果。这对于没有自己训练模型的用户，或者希望快速部署模型的用户非常有用。

  

**5.1 通过 Transformers 库使用推理 API**

  

Hugging Face 提供了一个简便的 pipeline API，用于快速进行推理。你只需要传入模型的名称和任务类型，pipeline 会自动选择合适的模型和 tokenizer，并进行推理。

  

**使用方法**：

• 假设你想进行文本生成：

  

from transformers import pipeline

  

generator = pipeline("text-generation", model="gpt2")

result = generator("Once upon a time", max_length=50)

print(result)

  

  

• 你可以指定其他任务，如文本分类、命名实体识别等：

  

classifier = pipeline("sentiment-analysis")

print(classifier("I love Hugging Face!"))

  

  

  

**5.2 通过 HTTP 调用推理 API**

  

如果你不想直接使用 transformers 库，也可以通过 HTTP 请求直接调用 Hugging Face 提供的 API。这对于开发 Web 服务或集成其他系统非常有用。

  

**步骤**：

1. 在 Hugging Face Hub 上找到你需要的模型，并获取 API Token。

2. 使用 HTTP 请求调用模型 API。

  

**示例**：

• 使用 curl 调用推理 API：

  

curl -X POST https://api-inference.huggingface.co/models/gpt2 \

     -H "Authorization: Bearer YOUR_API_TOKEN" \

     -d '{"inputs": "Once upon a time"}'

  

  

• 你也可以在 Python 中使用 requests 发送 POST 请求：

  

import requests

  

headers = {"Authorization": "Bearer YOUR_API_TOKEN"}

data = {"inputs": "Once upon a time"}

response = requests.post("https://api-inference.huggingface.co/models/gpt2", headers=headers, json=data)

print(response.json())

  

  

  

**5.3 推理 API 的高级功能**

• **批量推理**：你可以一次性发送多个输入进行批量推理。

• **自定义参数**：你可以根据需要调整推理参数，例如生成的文本长度、温度、top-k 采样等。


**6. Accelerate 库**

accelerate 是 Hugging Face 提供的一个用于加速模型训练和推理的库，特别适用于大规模训练任务。它简化了在多设备（如 GPU 和 TPU）上并行训练和分布式训练的复杂性。你可以通过它来轻松地在多个设备之间分配任务，提高计算效率。

  

**6.1 安装 Accelerate**

  

首先，你需要安装 accelerate 库：

  

pip install accelerate

  

**6.2 简化多设备训练**

  

accelerate 可以帮助你自动管理多 GPU 或多机器训练，并且可以通过简单的命令运行模型训练。它支持多种设备类型，包括 CPU、GPU 和 TPU。

  

**步骤**：

1. 在训练脚本中，首先导入 Accelerator 类：

  

from accelerate import Accelerator

  

  

2. 创建 Accelerator 实例并设置训练过程：

  

accelerator = Accelerator()

  

  

3. 使用 accelerator.prepare() 来自动准备模型、优化器和数据加载器：

  

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

  

  

4. 然后，你可以像平常一样进行训练，accelerate 会自动处理多设备的同步和调度。

  

**示例**：使用 accelerate 进行简单的训练

  

from accelerate import Accelerator

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW

from datasets import load_dataset

  

_# 初始化 Accelerator_

accelerator = Accelerator()

  

_# 加载数据集和模型_

dataset = load_dataset("imdb")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  

_# 准备数据_

train_dataset = dataset["train"].map(lambda x: tokenizer(x["text"], padding=True, truncation=True), batched=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8)

  

_# 初始化优化器_

optimizer = AdamW(model.parameters(), lr=5e-5)

  

_# 准备所有内容_

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

  

_# 训练过程_

for epoch in range(3):

    model.train()

    for batch in train_dataloader:

        optimizer.zero_grad()

        inputs = {key: batch[key].to(accelerator.device) for key in batch}

        labels = inputs.pop("label")

        outputs = model(**inputs, labels=labels)

        loss = outputs.loss

        accelerator.backward(loss)

        optimizer.step()

    print(f"Epoch {epoch} completed")

  

通过使用 accelerate，你可以轻松地在多设备上进行训练，不需要手动编写复杂的分布式训练代码。accelerate 处理了数据并行、梯度累积等方面的内容。

  

**6.3 并行推理**

  

accelerate 还支持推理时的多设备加速。例如，如果你有多个 GPU，并希望将推理负载分配到多个设备上，可以使用 accelerate 来加速推理。

  

from accelerate import Accelerator

from transformers import AutoModelForSequenceClassification, AutoTokenizer

  

accelerator = Accelerator()

  

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

  

_# 准备模型和 tokenizer_

model, tokenizer = accelerator.prepare(model, tokenizer)

  

_# 推理_

inputs = tokenizer("Hello, Hugging Face!", return_tensors="pt")

inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}

outputs = model(**inputs)

  

**7. Hugging Face Spaces**

  

Hugging Face Spaces 是一个用于托管和分享机器学习模型和应用的平台。你可以在 Hugging Face Spaces 上创建、部署和分享你的机器学习应用，支持通过 Streamlit、Gradio 等库快速搭建交互式界面。

  

**7.1 创建 Hugging Face Spaces**

  

Hugging Face Spaces 允许用户通过简单的代码和界面，快速构建 Web 应用来展示模型的推理能力。你可以选择使用 Gradio 或 Streamlit 来构建应用界面。

  

**步骤**：

1. 登录 Hugging Face 账户并创建一个 Space 页面。

2. 在 Space 中创建一个应用，可以选择 Gradio 或 Streamlit 框架。

  

**示例：使用 Gradio 创建交互式界面**：

  

pip install gradio

  

然后，创建一个简单的 Gradio 应用来展示模型：

  

import gradio as gr

from transformers import pipeline

  

_# 加载预训练模型_

classifier = pipeline("sentiment-analysis")

  

_# 创建 Gradio 界面_

def predict(text):

    return classifier(text)

  

_# 设置界面_

interface = gr.Interface(fn=predict, inputs="text", outputs="json")

  

_# 启动应用_

interface.launch()

  

这段代码创建了一个简单的文本分类应用，用户可以输入文本，应用会返回该文本的情感分析结果。

  

**7.2 使用 Streamlit 创建交互式界面**

  

Streamlit 也是一个非常流行的 Python 库，用于快速创建 Web 应用。你可以在 Hugging Face Spaces 中使用 Streamlit 构建应用。

  

pip install streamlit

  

然后，创建一个简单的应用：

  

import streamlit as st

from transformers import pipeline

  

_# 加载模型_

classifier = pipeline("sentiment-analysis")

  

_# 创建 Streamlit 应用_

st.title("Sentiment Analysis")

text_input = st.text_area("Enter text:")

  

if text_input:

    result = classifier(text_input)

    st.write(result)

  

然后，通过 streamlit run 启动应用：

  

streamlit run app.py

  

这样你就可以通过 Streamlit 创建一个简单的应用，用户输入文本后即可看到情感分析的结果。

  

**7.3 分享和部署应用**

  

一旦你创建了 Hugging Face Space，你可以分享它的链接给其他人，让他们也能访问和使用你的应用。Hugging Face 会为每个 Space 提供一个独立的 URL，你可以将其嵌入到文档、博客等地方。

• 在 Space 页面上，你可以选择公开或私有你的应用。如果你选择公开，其他人也可以访问并使用这个应用。

• 你也可以在应用中上传自己的数据或模型文件，使得应用更加丰富和个性化。

  

**8. Hugging Face 文档和社区支持**

  

Hugging Face 提供了丰富的文档和社区支持，帮助开发者解决问题并学习如何使用 Hugging Face 的工具。

  

**8.1 官方文档**

• [Hugging Face 官方文档](https://huggingface.co/docs)：包含了从模型加载、训练到部署等各个方面的详细说明，涵盖了各类 API 和功能的使用示例。

  

**8.2 Hugging Face 论坛**

• [Hugging Face 论坛](https://discuss.huggingface.co/)：你可以在这个论坛上向社区提问，分享经验，讨论最新的研究成果。论坛中有许多来自不同领域的专家，他们会回答你的问题，提供帮助。

  

**8.3 Hugging Face Discord**

• Hugging Face 还提供了一个 [Discord 频道](https://discord.gg/huggingface)，你可以在这里与其他开发者和研究人员实时交流。

  

**总结**

  

我们已经详细探讨了 Hugging Face 的许多功能，包括：

1. **Accelerate 库**：帮助在多个设备上加速训练和推理，简化分布式训练。

2. **Hugging Face Spaces**：用于创建和部署机器学习应用，支持 Gradio 和 Streamlit 界面，方便展示和分享模型。

3. **文档和社区支持**：提供丰富的文档、论坛和实时交流平台，帮助开发者解决问题。

  

通过使用 Hugging Face 提供的工具和平台，你可以高效地构建、训练、部署和分享机器学习模型。希望这些内容能对你有所帮助，如果有任何问题或需要进一步的示例，随时告诉我！