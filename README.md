#### 复现论文 

https://ieeexplore.ieee.org/document/9252123 （Self-Supervised Pretraining of Transformers for Satellite Image Time Series）

#### 前情提要：
遥感时序的一篇文章，主要做了两件事
1. 使用 bert-like 的思路，对时序进行自监督掩码训练（复现重点）
2. 文章中对预训练权重进行微调，适配分类任务（并不复现）

P.S. bert 的训练思路非常适合用作 encoder，而不是微调，由于剧情需要，趁机复刻这篇论文，并用于后续实验

#### 复刻过程

1. 自制时序数据集
通过 https://www.planet.com 进行时序数据集制作，区域为武汉襄阳，此处感谢 @Liuhai626
数据集截图如下：
<img width="1062" height="828" alt="dataset" src="https://github.com/user-attachments/assets/b12a2b7f-b4ca-49f5-9c55-a05c82f9f93e" />


3. 数据集处理

对时序数据集进行采样，将4个波段RGBNIR进行拼接，作为特征输入，见 data.csv

4. 模型构建
整体架构类似 bert，双向注意力，对时序进行掩码，监督目标是预测完整时序，最终训练学习

模型结构如下
```shell
SITSBERT(
  (embedding): ObservationEmbedding(
    (spectral_embed): Linear(in_features=4, out_features=64, bias=True)
  )
  (transformer): TransformerEncoder(
    (layers): ModuleList(
      (0-3): 4 x TransformerEncoderLayer(
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=128, out_features=128, bias=True)
        )
        (linear1): Linear(in_features=128, out_features=256, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (linear2): Linear(in_features=256, out_features=128, bias=True)
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.1, inplace=False)
        (dropout2): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (output_layer): Linear(in_features=128, out_features=4, bias=True)
)
```

5. 模型使用
① 作为 encoder 进行使用，负责将时序编码为向量，用于后续下游任务（分类分割检测等）

② 时序预测，回归任务，可视化如图

<img width="1521" height="391" alt="afa918591d78cef59f1c1ed6e9784ae5" src="https://github.com/user-attachments/assets/b86cb6cf-1e1c-4dc4-a52c-5060bd6064d1" />

#### 小组成员：

赵昌浩（2025303120178）
陈星灿（2025303110131）
李世豪（2025303120170）
李迁（2025303120110）
