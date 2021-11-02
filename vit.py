import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class CnnPatchEmbed(nn.Module):
    def __init__(self, patch_size, emb_dim=512):
        super().__init__()
        self.to_patch_embedding = nn.Sequential(
            torch.nn.Conv2d(3, emb_dim, patch_size, patch_size),
            Rearrange('b c p1 p2 -> b (p1 p2) c'),
        )

    def forward(self, x):
        return self.to_patch_embedding(x)


class LinearPatchEmbed(nn.Module):
    def __init__(self, patch_size, emb_dim=512):
        super().__init__()
        patch_dim = 3 * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, emb_dim),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, n_heads=12, proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, _ = x.shape

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (q @ k_t) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    def __init__(self, in_out_features, hidden_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_out_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0, p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, proj_p=p)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_out_features=dim, hidden_features=hidden_features)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=384, patch_size=16, n_classes=1000,
                 emb_dim=768, depth=12, n_heads=12, mlp_ratio=4., p=0.):
        super().__init__()

        self.patch_embed = CnnPatchEmbed(
            patch_size=patch_size,
            emb_dim=emb_dim,
        )
        n_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + n_patches, emb_dim))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [Block(dim=emb_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, p=p) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(n_samples, -1, -1)  # (n_samples, 1, emb_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, emb_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, emb_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x
