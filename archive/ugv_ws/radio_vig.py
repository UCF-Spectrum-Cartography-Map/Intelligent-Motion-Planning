"""
radio_vig.py — exact ViG reconstruction model for GNN_best.pth.

Extracted verbatim from the training notebook (cell 2). Loads with strict=True.

    input : (B, 3, 256, 256) = [ch0, ch1, sparse_path_loss]
    output: (B, 1, 256, 256) = dense normalized path-loss map (low = emitter)

NOTE on the input: in the notebook version that defines this GNN, build_wnet_input
feeds [zeros, zeros, observed_pl] — the building mask is NOT given to the model.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_graph(x, k):
    """k-NN graph in feature space. x:(B,C,N,1) -> (B,N,k) neighbour indices."""
    B, C, N, _ = x.shape
    x_sq = x.squeeze(-1).permute(0, 2, 1)               # (B,N,C)
    dist = torch.cdist(x_sq, x_sq, p=2)                 # (B,N,N)
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    return idx[:, :, 1:]                                # drop self-loop


class DynConv2d(nn.Module):
    """EdgeConv-style dynamic graph conv on (B,C,N,1)."""
    def __init__(self, in_channels, out_channels, k=9):
        super().__init__()
        self.k    = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1)

    def forward(self, x):
        B, C, N, _ = x.shape
        idx       = knn_graph(x, self.k)
        x_flat    = x.squeeze(-1).permute(0, 2, 1)
        idx_exp   = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        neighbors = torch.gather(
            x_flat.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx_exp)
        center = x_flat.unsqueeze(2).expand_as(neighbors)
        edge   = torch.cat([center, neighbors - center], dim=-1)
        agg    = edge.max(dim=2).values
        agg    = agg.permute(0, 2, 1).unsqueeze(-1)
        return self.conv(agg)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep) * rand


class GrapherModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=9, drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
                                 nn.BatchNorm2d(in_channels))
        self.graph_conv = nn.Sequential(DynConv2d(in_channels, hidden_channels, k),
                                        nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_channels, in_channels, 1),
                                 nn.BatchNorm2d(in_channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x); x = self.graph_conv(x); x = self.fc2(x)
        return self.drop_path(x) + shortcut


class FFNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 1),
                                 nn.BatchNorm2d(hidden_channels), nn.GELU())
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_channels, in_channels, 1),
                                 nn.BatchNorm2d(in_channels))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x); x = self.fc2(x)
        return self.drop_path(x) + shortcut


class ViGBlock(nn.Module):
    def __init__(self, channels, k=9, drop_path=0.0):
        super().__init__()
        self.grapher = GrapherModule(channels, channels * 2, k, drop_path)
        self.ffn     = FFNModule(channels, channels * 4, drop_path)

    def forward(self, x):
        return self.ffn(self.grapher(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, E, n_h, n_w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2).reshape(B, E, n_h, n_w)


class PatchUnembed(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size=8):
        super().__init__()
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class ViGPatchModel(nn.Module):
    """Vision GNN (ViG): (B,3,H,W) -> (B,1,H,W) dense path-loss."""
    def __init__(self, in_channels=3, out_channels=1, embed_dim=64,
                 num_blocks=4, patch_size=8, k=9, drop_path=0.1):
        super().__init__()
        self.patch_size  = patch_size
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        dpr = [v.item() for v in torch.linspace(0, drop_path, num_blocks)]
        self.blocks = nn.ModuleList(
            [ViGBlock(embed_dim, k=k, drop_path=dpr[i]) for i in range(num_blocks)])
        self.norm          = nn.BatchNorm2d(embed_dim)
        self.patch_unembed = PatchUnembed(embed_dim, out_channels, patch_size)

    def _pad_to_patch(self, x):
        B, C, H, W = x.shape
        ph = (self.patch_size - H % self.patch_size) % self.patch_size
        pw = (self.patch_size - W % self.patch_size) % self.patch_size
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))
        return x, H, W

    def forward(self, x):
        x, orig_H, orig_W = self._pad_to_patch(x)
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        _, E, n_h, n_w = x.shape
        x = x.reshape(B, E, n_h * n_w, 1)
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(B, E, n_h, n_w)
        x = self.norm(x)
        x = self.patch_unembed(x)
        return x[:, :, :orig_H, :orig_W]


def load_gnn(weights_path, device="cpu"):
    model = ViGPatchModel(in_channels=3, out_channels=1, embed_dim=64,
                          num_blocks=4, patch_size=8, k=9, drop_path=0.1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))  # strict
    model.eval()
    return model


if __name__ == "__main__":
    import sys
    m = load_gnn(sys.argv[1] if len(sys.argv) > 1 else "GNN_best.pth")
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        y = m(x)
    print("ok — output", tuple(y.shape), "min/max", float(y.min()), float(y.max()))
