# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from inspect import isfunction
from functools import partial
from einops import rearrange
import matplotlib.pyplot as plt

def exists(x): return x is not None

def default(val, d):
    if exists(val): return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim): return nn.ConvTranspose2d(dim, dim, 4, 2, 1)
def Downsample(dim): return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


class ConditionalUnet(nn.Module):
    def __init__(self, dim, out_dim=3, channels=4, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.channels = channels
        init_dim = dim // 3 * 2
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ConvNextBlock, mult=2)

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        self.time_mlp_h = nn.Sequential(SinusoidalPositionEmbeddings(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x, time, h_time=None):
        x = self.init_conv(x)
        t = self.time_mlp(time)
        if h_time is not None:
            t = t + self.time_mlp_h(h_time)

        h_list = []
        for b1, b2, down in self.downs:
            x = b1(x, t)
            x = b2(x, t)
            h_list.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for b1, b2, up in self.ups:
            x = torch.cat((x, h_list.pop()), dim=1)
            x = b1(x, t)
            x = b2(x, t)
            x = up(x)
        return self.final_conv(x)


class SAR2RGBDataset(Dataset):
    def __init__(self, sar_dir, rgb_dir, image_size=256):
        self.sar_dir = sar_dir
        self.rgb_dir = rgb_dir
        self.file_names = sorted([f for f in os.listdir(sar_dir) if f.endswith('.png')])
        
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        sar_img = Image.open(os.path.join(self.sar_dir, fname)).convert('L')
        rgb_img = Image.open(os.path.join(self.rgb_dir, fname)).convert('RGB')
        return self.transform(sar_img), self.transform(rgb_img)


class ConditionalMeanFlowLoss:
    def __init__(self, P_mean=-0.4, P_std=1.0, data_proportion=0.75, norm_p=1.0, norm_eps=1.0):
        self.P_mean, self.P_std = P_mean, P_std
        self.data_proportion = data_proportion
        self.norm_p, self.norm_eps = norm_p, norm_eps

    def __call__(self, net, sar, rgb):
        device = sar.device
        batch_size = sar.shape[0]
        shape = (batch_size, 1, 1, 1)

        t = torch.sigmoid(torch.randn(shape, device=device) * self.P_std + self.P_mean)
        r = torch.sigmoid(torch.randn(shape, device=device) * self.P_std + self.P_mean)
        t, r = torch.max(t, r), torch.min(t, r)
        
        zero_mask = (torch.arange(batch_size, device=device) < int(batch_size * self.data_proportion)).view(shape)
        r = torch.where(zero_mask, t, r)

        y = rgb
        n = torch.randn_like(y)
        z_t = (1 - t) * y + t * n
        v_g = n - y

        def u_wrapper(z, time, r_val, cond):
            # 将 SAR 条件与当前状态拼接
            net_input = torch.cat([z, cond], dim=1)
            return net(net_input, time.squeeze(), h_time=(time - r_val).squeeze())

        primals = (z_t, t, r)
        tangents = (v_g, torch.ones_like(t), torch.zeros_like(t))
        u, du_dt = torch.func.jvp(lambda z, tm, rv: u_wrapper(z, tm, rv, sar), primals, tangents)

        u_tgt = (v_g - torch.clamp(t - r, min=0.0, max=1.0) * du_dt).detach()
        unweighted_loss = (u - u_tgt).pow(2).sum(dim=[1, 2, 3])
        
        with torch.no_grad():
            adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)
        
        return (unweighted_loss * adaptive_weight).mean()


@torch.no_grad()
def generate_1step(model, sar_img):
    model.eval()
    device = sar_img.device
    # 1. 采样初始噪声
    noise = torch.randn(sar_img.shape[0], 3, sar_img.shape[2], sar_img.shape[3]).to(device)
    # 2. 拼接条件
    net_input = torch.cat([noise, sar_img], dim=1)
    # 3. 设置时间为 1.0 (完全噪声状态)
    t = torch.ones(sar_img.shape[0]).to(device)
    # 4. 预测速度并一步回退到 x0
    v_pred = model(net_input, t, h_time=t)
    x0 = noise - v_pred
    return x0.clamp(-1, 1)


def main():
    SAR_DIR = './data/sar'     
    RGB_DIR = './data/rgb'
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 100
    LR = 2e-4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SAR2RGBDataset(SAR_DIR, RGB_DIR, image_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = ConditionalUnet(dim=64, out_dim=3, channels=4, dim_mults=(1, 2, 4, 8)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = ConditionalMeanFlowLoss()

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader)
        for sar, rgb in pbar:
            sar, rgb = sar.to(DEVICE), rgb.to(DEVICE)
            
            loss = loss_fn(model, sar, rgb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        # 每个 Epoch 结束后保存一次并预览
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"mf_sar2rgb_epoch_{epoch}.pt")
            # 预览第一个样本
            test_sar = sar[0:1]
            test_gt = rgb[0:1]
            gen_rgb = generate_1step(model, test_sar)
            
            # 可视化 (反归一化)
            plt.figure(figsize=(10, 4))
            plt.subplot(1,3,1); plt.imshow(test_sar[0,0].cpu()*0.5+0.5, cmap='gray'); plt.title("SAR")
            plt.subplot(1,3,2); plt.imshow((gen_rgb[0].cpu().permute(1,2,0)*0.5+0.5).numpy()); plt.title("Generated")
            plt.subplot(1,3,3); plt.imshow((test_gt[0].cpu().permute(1,2,0)*0.5+0.5).numpy()); plt.title("GT")
            plt.show()

if __name__ == "__main__":
    main()
