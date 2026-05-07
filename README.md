#### 复现论文 

https://ieeexplore.ieee.org/document/9252123 （Self-Supervised Pretraining of Transformers for Satellite Image Time Series Classification）

#### 前情提要：
遥感时序的一篇文章，主要做了两件事
1. 使用 bert-like 的思路，对时序进行自监督掩码训练
2. 在此基础上进行分类

#### 复刻过程

1. 自制数据集
通过 https://www.planet.com 进行时序数据集制作，区域为武汉襄阳，此处感谢 @Liuhai626
数据集截图如下：

2. 数据集处理
对时序数据集进行采样，将4个波段RGBNIR进行拼接，作为特征输入，见 data.csv

3. 模型构建
整体架构类似 bert，双向注意力，对时序进行掩码，监督目标是预测完整时序，最终训练学习

4. 模型使用
作为 encoder 进行使用，负责将时序编码为向量，最终适配下游任务（分类分割检测等）

#### 小组成员：

赵昌浩（2025303120178）
陈星灿（2025303110131）
李世豪（2025303120170）
李迁（2025303120110）