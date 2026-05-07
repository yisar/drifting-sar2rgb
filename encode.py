import torch
import torch.nn as nn
import numpy as np

# 1. 必须保留与训练时一致的类定义
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
        spectral_embed = self.spectral_embed(spectral_data)
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
        # 预训练时的输出层
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, spectral_data, doy, return_encoded=True):
        # 得到嵌入表示 [Batch, 6, 128]
        x = self.embedding(spectral_data, doy)
        # 经过 Transformer 编码 [Batch, 6, 128]
        encoded = self.transformer(x)
        
        if return_encoded:
            # 这里的池化操作将 (Batch, 6, 128) 变为 (Batch, 128)
            # 这就是你想要的“编码后的向量”
            return torch.mean(encoded, dim=1)
        
        return self.output_layer(encoded)

# ==========================================
# 2. 加载模型并编码
# ==========================================

def load_and_encode(model_path, csv_sample_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化相同的模型
    model = SITSBERT(input_dim=4, embed_dim=128, num_heads=8, num_layers=4, ff_dim=256).to(device)

    print(model)
    
    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到推理模式
    print(f"成功加载模型: {model_path}")

    # 准备输入数据 (假设输入 1 个样本)
    # 输入应归一化，形状 [1, 6, 4]
    input_tensor = torch.tensor(csv_sample_data, dtype=torch.float32).unsqueeze(0).to(device) / 10000.0
    # 对应的 5-10月 DOY
    doy_tensor = torch.tensor([[135, 166, 196, 227, 258, 288]], dtype=torch.long).to(device)

    with torch.no_grad():
        # 得到 128 维特征向量
        feature_vector = model(input_tensor, doy_tensor, return_encoded=True)

    return feature_vector

# --- 测试运行 ---
if __name__ == "__main__":
    # 这里填入你的模型文件名
    WEIGHT_PATH = "sits_bert_may_oct_v2.pth"
    
    # 模拟从 CSV 提取的一条 [6, 4] 数据
    sample_data = np.random.uniform(500, 3000, (6, 4)).tolist() 
    
    vector = load_and_encode(WEIGHT_PATH, sample_data)
    
    print("-" * 30)
    print(f"编码后的向量结构: {vector.shape}")
    print(f"向量数据类型: {vector.dtype}")
    print(f"向量前5位值: {vector[0, :5].cpu().numpy()}")
    print("-" * 30)