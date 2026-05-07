import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------
# 1. 专门用于推理的类：增加注意力提取功能
# ---------------------------------------------------------
class SITSBERTInference(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.embedding = original_model.embedding
        self.layers = original_model.transformer.layers
        self.output_layer = original_model.output_layer

    def forward(self, spectral_data, doy):
        x = self.embedding(spectral_data, doy)
        
        all_attentions = []
        # 手动遍历每一层，获取 MultiheadAttention 的输出
        for layer in self.layers:
            # PyTorch 的 MultiheadAttention 在 need_weights=True 时返回 (output, weights)
            # 注意：TransformerEncoderLayer 默认不返回 weights，我们直接调用它的 self_attn
            attn_output, attn_weights = layer.self_attn(x, x, x, need_weights=True)
            x = layer._ff_block(layer.norm1(x + attn_output)) # 简化版 residual
            all_attentions.append(attn_weights)
        
        reconstructed = self.output_layer(x)
        return reconstructed, all_attentions

# ---------------------------------------------------------
# 2. 核心可视化函数
# ---------------------------------------------------------
def run_visual_demo(model_path, sample_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载原始模型
    from train import SITSBERT # 确保能导入你的类
    base_model = SITSBERT(4, 128, 8, 4, 256)
    base_model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 包装为推理模型
    model = SITSBERTInference(base_model).to(device).eval()

    # 准备输入：模拟 Mask 掉 7月 (index 2) 和 8月 (index 3)
    original = torch.tensor(sample_data, dtype=torch.float32).unsqueeze(0).to(device) / 10000.0
    doy = torch.tensor([[135, 166, 196, 227, 258, 288]], dtype=torch.long).to(device)
    
    masked_input = original.clone()
    masked_indices = [2, 3] # 7, 8月
    masked_input[:, masked_indices, :] = 0.0

    # 推理
    with torch.no_grad():
        reconstructed, attentions = model(masked_input, doy)

    # --- 绘图 1: 时序重建对比 ---
    months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    bands = ['Red', 'Green', 'Blue', 'NIR']
    
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.plot(months, original[0, :, i].cpu(), 'go-', label='Truth', alpha=0.4)
        plt.plot(months, reconstructed[0, :, i].cpu(), 'rx--', label='Pred')
        plt.title(f'{bands[i]} Band')
        if i == 0: plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 绘图 2: 注意力热力图 ---
    # 我们取最后一层 Transformer 的注意力权重 [Batch, Time, Time]
    last_layer_attn = attentions[-1][0].cpu().numpy() 

    plt.figure(figsize=(8, 6))
    sns.heatmap(last_layer_attn, annot=True, fmt=".2f", cmap='YlGnBu',
                xticklabels=months, yticklabels=months)
    plt.title("Self-Attention Weights (Last Layer)")
    plt.xlabel("Key (Month Looked At)")
    plt.ylabel("Query (Month Being Processed)")
    plt.show()

# 模拟一条数据测试
sample_pixel = np.random.uniform(500, 4000, (6, 4)).tolist()
run_visual_demo("sits_bert.pth", sample_pixel)