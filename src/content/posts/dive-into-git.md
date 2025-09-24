---
title: dive-into-git
description: 怎么使用git？不仅限于简单使用
published: 2024-11-11
tags:
  - git
category: DEVTools
---
git，可以说既熟悉又陌生。大家日常使用可能仅限于个人开发，即使用`git clone url`, `git add .`, `git commit -m 'waibiwaibi'`等。但是如果涉及到 




```bash
// 只git下来近十次的信息
git clone git@github.com:duffy/example.git --depth 10
// 更新分支模块
git submodule update --init --recursive

git status


git config --global user.name "duffy"
git config --global user.email "duffy@qq.com"
```

如果使用ssh连接那么就要配置密钥。
本地生成密钥，一个公钥、一个私钥。公钥存到GitHub平台上。

