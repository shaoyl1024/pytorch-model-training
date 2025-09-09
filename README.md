#  PyTorch Model Training Engine

# 项目概述

本项目基于 Pytorch 开发稳健的预测模型，用于估计用户点击所展示广告的概率。准确的 CTR 预测对于优化广告投放、提升用户体验以及最大化平台和广告商的广告收入至关重要。

# 项目结构

```text
pytorch-model-training/
├── data/                # 项目数据
├── src/                 # 源代码和工具函数
│   ├── features/                     # 特征工程（数据分析、预处理）
│   ├── figures/                      # 生成的可视化图表
│   ├── models/                       # 模型训练
│   ├── notebooks/                    # Jupyter笔记本，用于交互式数据分析和实验
│   ├── pytorch/                
│   │   ├── ctr                       # CTR训练相关文件
│   │       ├── 202509v1/             # 202509v1版本模型文件
│   │           ├── criteo_ctr_model.onnx     # 模型文件（ONNX格式）
│   │           ├── criteo_preprocessor.json  # 预处理器参数（JSON格式）
│   │           └── ctr_train_v2.py           # 执行训练的代码
├── figures/             # 生成的可视化图表
├── utils/               # 工具包
├── requirements.txt     # 项目依赖清单（含版本锁定）
├── .gitignore           # 版本控制中需排除的文件/文件夹
└── README.md            # 项目文档（本文档）
```
