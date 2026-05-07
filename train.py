import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ObservationEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(ObservationEmbedding, self).__init__()
        self.embed_dim = embed_dim
        # 线性层将光谱波段映射到 embed_dim 的一半
        self.spectral_embed = nn.Linear(input_dim, embed_dim // 2)
        # 预生成 366 天的 Positional Encoding (PE)
        self.register_buffer('positional_encoding', self._generate_positional_encoding())

    def _generate_positional_encoding(self):
        position = torch.arange(0, 366).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim // 2, 2) * -(torch.log(torch.tensor(10000.0)) / (self.embed_dim // 2)))
        pe = torch.zeros(366, self.embed_dim // 2)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, spectral_data, doy):
        # spectral_data: [Batch, Time, Bands] -> [128, 6, 4]
        # spectral_embed: [Batch, Time, embed_dim // 2] -> [128, 6, 64]
        spectral_embed = self.spectral_embed(spectral_data)
        
        # doy 形状应为 [Batch, Time] -> [128, 6]
        # 索引后的 pe 形状为 [Batch, Time, embed_dim // 2] -> [128, 6, 64]
        pe = self.positional_encoding[doy.long()]
        
        return torch.cat([spectral_embed, pe], dim=-1)

class SITSBERT(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim):
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
        
        # 5, 6, 7, 8, 9, 10 月对应的 DOY
        self.doy_values = torch.tensor([135, 166, 196, 227, 258, 288], dtype=torch.long)
        
        all_samples = []
        for _, row in df.iterrows():
            sample_time_series = []
            for t in range(1, num_timesteps + 1):
                pixel_at_t = [row[f'r{t}'], row[f'g{t}'], row[f'b{t}'], row[f'nir{t}']]
                sample_time_series.append(pixel_at_t)
            all_samples.append(sample_time_series)
            
        # 归一化反射率
        self.data = torch.tensor(all_samples, dtype=torch.float32) / 10000.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.doy_values

def apply_mask(spectral_data, mask_ratio=0.15):
    batch_size, num_timesteps, num_bands = spectral_data.shape
    device = spectral_data.device
    rand_matrix = torch.rand(batch_size, num_timesteps, device=device)
    mask = (rand_matrix < mask_ratio).float()
    masked_data = spectral_data.clone()
    masked_data[mask == 1] = 0.0
    return masked_data, mask

def train_sits_bert(csv_file_path):
    # 超参数
    BATCH_SIZE = 128
    EPOCHS = 100
    EMBED_DIM = 128 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SITSDatasetCSV(csv_file_path)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SITSBERT(input_dim=4, embed_dim=EMBED_DIM, num_heads=8, num_layers=4, ff_dim=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss(reduction='none')

    print(f"开始训练，设备: {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for spectral_data, doy_batch in dataloader:
            # spectral_data 形状: [Batch, 6, 4]
            
            spectral_data = spectral_data.to(DEVICE)
            doy_batch = doy_batch.to(DEVICE)
            
            # 如果 batch 最后一组不满，确保 doy 形状正确
            if doy_batch.shape[0] != spectral_data.shape[0]:
                doy_batch = doy_batch[:spectral_data.shape[0]]

            # 掩码
            masked_input, mask = apply_mask(spectral_data, mask_ratio=0.15)
            
            # 此时传入的 doy_batch 形状为 [Batch, 6]，模型内索引后得到 [Batch, 6, 64]
            output = model(masked_input, doy_batch)
            
            # 损失计算 (仅针对被 mask 的部分)
            loss_matrix = criterion(output, spectral_data)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss_matrix)
            masked_loss = (loss_matrix * mask_expanded).sum() / (mask_expanded.sum() + 1e-8)
            
            optimizer.zero_grad()
            masked_loss.backward()
            optimizer.step()
            
            total_loss += masked_loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), "sits_bert_may_oct_v2.pth")
    print("模型保存成功。")

if __name__ == "__main__":
    train_sits_bert("combined_data.csv")