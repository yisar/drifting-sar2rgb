import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------------------------------------------------------
# 1. 模型定义 (需与训练时完全一致)
# ---------------------------------------------------------
class ObservationEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.spectral_embed = nn.Linear(input_dim, embed_dim // 2)
        self.register_buffer('positional_encoding', self._generate_positional_encoding())

    def _generate_positional_encoding(self):
        position = torch.arange(0, 366).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (self.embed_dim // 2)))
        pe = torch.zeros(366, self.embed_dim // 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, spectral_data, doy):
        spectral_embed = self.spectral_embed(spectral_data)
        pe = self.positional_encoding[doy.long()]
        return torch.cat([spectral_embed, pe], dim=-1)

class SITSBERT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim):
        super().__init__()
        self.embedding = ObservationEmbedding(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, spectral_data, doy):
        x = self.embedding(spectral_data, doy)
        x = self.transformer(x)
        return self.output_layer(x)

# ---------------------------------------------------------
# 2. 从 CSV 提取特定样本的函数
# ---------------------------------------------------------
def get_sample_from_csv(csv_path, row_index=0):
    """
    从 CSV 中读取一行，并转换为 [6, 4] 结构
    """
    df = pd.read_csv(csv_path)
    # 选取一行
    row = df.iloc[row_index]
    
    # 根据你的列名规则提取: r1, g1, b1, nir1 ... r6, g6, b6, nir6
    sample_list = []
    for t in range(1, 7):
        pixel_at_t = [row[f'r{t}'], row[f'g{t}'], row[f'b{t}'], row[f'nir{t}']]
        sample_list.append(pixel_at_t)
    
    print(f"已选取 CSV 第 {row_index} 行数据 (ID: {row.get('id', 'N/A')})")
    return sample_list

# ---------------------------------------------------------
# 3. 推理与可视化逻辑
# ---------------------------------------------------------
def run_visual_demo(csv_path, model_path, row_idx=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 获取真实数据
    raw_data = get_sample_from_csv(csv_path, row_index=row_idx)
    original = torch.tensor(raw_data, dtype=torch.float32).unsqueeze(0).to(device) / 10000.0
    doy = torch.tensor([[135, 166, 196, 227, 258, 288]], dtype=torch.long).to(device)

    # 2. 加载模型
    model = SITSBERT(4, 128, 8, 4, 256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. 模拟 Mask (比如遮盖 7, 8月)
    masked_input = original.clone()
    masked_indices = [2, 3] 
    masked_input[:, masked_indices, :] = 0.0

    # 4. 前向传播
    with torch.no_grad():
        reconstructed = model(masked_input, doy)

    # 5. 绘图
    months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    bands = ['Red', 'Green', 'Blue', 'NIR']
    colors = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6']
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for i in range(4):
        axes[i].plot(months, original[0, :, i].cpu(), 'o-', label='Ground Truth', color='gray', alpha=0.4)
        axes[i].plot(months, reconstructed[0, :, i].cpu(), 's--', label='Reconstructed', color=colors[i])
        
        # 把被遮盖的点特别标出来
        axes[i].scatter([months[j] for j in masked_indices], 
                        original[0, masked_indices, i].cpu(), 
                        color='black', marker='x', s=80, zorder=5)
        
        axes[i].set_title(f'Band: {bands[i]}')
        if i == 0: axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f"Real Data Reconstruction (CSV Row: {row_idx})", fontsize=14)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 执行
# ---------------------------------------------------------
if __name__ == "__main__":
    CSV_FILE = "data.csv"
    MODEL_FILE = "sits_bert.pth"
    
    # 你可以更改 row_idx 来查看不同地块的效果
    run_visual_demo(CSV_FILE, MODEL_FILE, row_idx=5)