import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

class ObservationEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(ObservationEmbedding, self).__init__()
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
        # spectral_data: [Batch, Time, Bands]
        spectral_embed = self.spectral_embed(spectral_data)
        pe = self.positional_encoding[doy.long()]
        return torch.cat([spectral_embed, pe], dim=-1)

class SITSBERT(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, num_heads=8, num_layers=4, ff_dim=256):
        super(SITSBERT, self).__init__()
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


class SITSDatasetCSV(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        num_timesteps = 6
        self.doy_values = torch.tensor([135, 166, 196, 227, 258, 288], dtype=torch.long)
        
        all_samples = []
        for _, row in df.iterrows():
            sample_time_series = []
            for t in range(1, num_timesteps + 1):
                # 核心修复：确保每个时间步抓取正确的 R, G, B, NIR
                pixel_at_t = [row[f'r{t}'], row[f'g{t}'], row[f'b{t}'], row[f'nir{t}']]
                sample_time_series.append(pixel_at_t)
            all_samples.append(sample_time_series)
            
        self.data = torch.tensor(all_samples, dtype=torch.float32) / 10000.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.doy_values


def run_visual_demo(csv_path, model_path, row_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载数据 (使用与训练一致的逻辑)
    dataset = SITSDatasetCSV(csv_path)
    original, doy = dataset[row_idx]
    original = original.unsqueeze(0).to(device) # [1, 6, 4]
    doy = doy.unsqueeze(0).to(device)           # [1, 6]

    # 2. 加载模型
    model = SITSBERT().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("模型加载成功！")
    except:
        print("未找到模型文件，将使用随机权重展示（仅测试结构）")

    model.eval()

    # 3. 模拟遮盖 (遮盖 7月和 8月)
    masked_input = original.clone()
    masked_indices = [2, 3] 
    masked_input[:, masked_indices, :] = 0.0

    # 4. 推理
    with torch.no_grad():
        reconstructed = model(masked_input, doy)

    # 5. 绘图
    months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    bands = ['Red', 'Green', 'Blue', 'NIR']
    colors = ['red', 'green', 'blue', 'purple']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        # 转换回原始量级打印对比 (乘以 10000)
        true_val = original[0, :, i].cpu() * 10000
        pred_val = reconstructed[0, :, i].cpu() * 10000
        
        axes[i].plot(months, true_val, 'o-', label='Ground Truth', color='gray', alpha=0.3)
        axes[i].plot(months, pred_val, 's--', label='SITS-BERT Pred', color=colors[i])
        
        # 标记 Mask 点
        axes[i].scatter([months[j] for j in masked_indices], 
                        true_val[masked_indices], 
                        color='black', marker='x', s=100, label='Masked', zorder=5)
        
        axes[i].set_title(f'Band: {bands[i]}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.2)

    plt.suptitle(f"Reconstruction Visualization (Row Index: {row_idx})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    CSV_FILE = "data.csv"
    MODEL_FILE = "sits_bert.pth"
    
    # 如果你想训练，取消下面这一行的注释：
    # train_sits_bert(CSV_FILE) 
    
    # 运行可视化
    run_visual_demo(CSV_FILE, MODEL_FILE, row_idx=0)
