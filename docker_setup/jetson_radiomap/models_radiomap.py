"""
models_radiomap.py
==================
Model architectures for the radio-map reconstruction ablation study,
extracted verbatim (architecture-wise) from `ablation_study_complete (1).ipynb`
so that the saved `*_best.pth` state-dicts can be re-loaded outside of Colab
on the Jetson Orin Nano.

Only the four models you intend to benchmark are exported here:
    - SpectrumCNN      ("CNN_best.pth")
    - RadioWNet        ("WNet_best.pth")   <- note: notebook uses "WNet", not "WNET"
    - PartialConvMAE   ("PartialConvMAE_best.pth")
    - ViGPatchModel    ("GNN_best.pth")

All four take  inputs of shape (B, 3, 256, 256):
    channel 0 -> building_mask  (0/1)
    channel 1 -> tx_origin      (0/1)
    channel 2 -> sparse_path_loss_norm  ([0,1], zeros where unknown)
and produce a (B, 1, 256, 256) dense path-loss estimate (normalised to [0,1]).

IMPORTANT export note for RadioWNet:
    The training forward() returns a tuple (out1, out2).  For inference /
    export we only care about the refined output `out2`.  Use
    `WNetExportWrapper` (defined below) so the exported graph has a single
    tensor output.

Config constants mirror the notebook so normalisation is consistent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
#  Config (mirrors the notebook)
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 256
MIN_DB = 30.0
MAX_DB = 140.0

# GNN hyper-params (must match how GNN_best.pth was trained)
GNN_EMBED_DIM = 64
GNN_NUM_BLOCKS = 4
GNN_PATCH_SIZE = 8
GNN_K = 9
GNN_DROP_PATH = 0.1

MODEL_COLORS = {
    "CNN": "#4C72B0",
    "WNet": "#DD8452",
    "PartialConvMAE": "#C44E52",
    "GNN": "#8172B2",
}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
#  1. CNN  — SpectrumCNN
# ══════════════════════════════════════════════════════════════════════════════
class SpectrumCNN(nn.Module):
    """Encoder-decoder CNN. Input (B,3,H,W) -> output (B,1,H,W)."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(self.up3(b))
        d2 = self.dec2(self.up2(d3))
        d1 = self.dec1(self.up1(d2))
        return self.head(d1)


# ══════════════════════════════════════════════════════════════════════════════
#  2. WNet — RadioWNet (double U-Net)
# ══════════════════════════════════════════════════════════════════════════════
def convrelu(in_ch, out_ch, k, pad, pool):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=pad),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool),
    )


def convreluT(in_ch, out_ch, k, pad):
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, k, stride=2, padding=pad),
        nn.ReLU(inplace=True),
    )


class RadioWNet(nn.Module):
    """Double U-Net. forward() returns (out1, out2) -> use out2 for inference."""

    def __init__(self, inputs=3):
        super().__init__()
        # ── 1st U-Net ──
        self.layer00 = convrelu(inputs, 6, 3, 1, 1)
        self.layer0 = convrelu(6, 40, 5, 2, 2)
        self.layer1 = convrelu(40, 50, 5, 2, 2)
        self.layer10 = convrelu(50, 60, 5, 2, 1)
        self.layer2 = convrelu(60, 100, 5, 2, 2)
        self.layer20 = convrelu(100, 100, 3, 1, 1)
        self.layer3 = convrelu(100, 150, 5, 2, 2)
        self.layer4 = convrelu(150, 300, 5, 2, 2)
        self.layer5 = convrelu(300, 500, 5, 2, 2)
        self.conv_up5 = convreluT(500, 300, 4, 1)
        self.conv_up4 = convreluT(600, 150, 4, 1)
        self.conv_up3 = convreluT(300, 100, 4, 1)
        self.conv_up20 = convrelu(200, 100, 3, 1, 1)
        self.conv_up2 = convreluT(200, 60, 6, 2)
        self.conv_up10 = convrelu(120, 50, 5, 2, 1)
        self.conv_up1 = convreluT(100, 40, 6, 2)
        self.conv_up0 = convreluT(80, 20, 6, 2)
        self.conv_up00 = convrelu(20 + 6 + inputs, 20, 5, 2, 1)
        self.conv_up000 = convrelu(20 + inputs, 1, 5, 2, 1)
        # ── 2nd U-Net ──
        self.Wlayer00 = convrelu(inputs + 1, 20, 3, 1, 1)
        self.Wlayer0 = convrelu(20, 30, 5, 2, 2)
        self.Wlayer1 = convrelu(30, 40, 5, 2, 2)
        self.Wlayer10 = convrelu(40, 50, 5, 2, 1)
        self.Wlayer2 = convrelu(50, 60, 5, 2, 2)
        self.Wlayer20 = convrelu(60, 70, 3, 1, 1)
        self.Wlayer3 = convrelu(70, 90, 5, 2, 2)
        self.Wlayer4 = convrelu(90, 110, 5, 2, 2)
        self.Wlayer5 = convrelu(110, 150, 5, 2, 2)
        self.Wconv_up5 = convreluT(150, 110, 4, 1)
        self.Wconv_up4 = convreluT(220, 90, 4, 1)
        self.Wconv_up3 = convreluT(180, 70, 4, 1)
        self.Wconv_up20 = convrelu(140, 60, 3, 1, 1)
        self.Wconv_up2 = convreluT(120, 50, 6, 2)
        self.Wconv_up10 = convrelu(100, 40, 5, 2, 1)
        self.Wconv_up1 = convreluT(80, 30, 6, 2)
        self.Wconv_up0 = convreluT(60, 20, 6, 2)
        self.Wconv_up00 = convrelu(20 + 20 + inputs + 1, 20, 5, 2, 1)
        self.Wconv_up000 = convrelu(20 + inputs + 1, 1, 5, 2, 1)

    def _unet1(self, x):
        l00 = self.layer00(x)
        l0 = self.layer0(l00)
        l1 = self.layer1(l0)
        l10 = self.layer10(l1)
        l2 = self.layer2(l10)
        l20 = self.layer20(l2)
        l3 = self.layer3(l20)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        u = self.conv_up5(l5)
        u = self.conv_up4(torch.cat([u, l4], 1))
        u = self.conv_up3(torch.cat([u, l3], 1))
        u = self.conv_up20(torch.cat([u, l20], 1))
        u = self.conv_up2(torch.cat([u, l2], 1))
        u = self.conv_up10(torch.cat([u, l10], 1))
        u = self.conv_up1(torch.cat([u, l1], 1))
        u = self.conv_up0(torch.cat([u, l0], 1))
        u = torch.cat([u, l00, x], 1)
        u = self.conv_up00(u)
        u = self.conv_up000(torch.cat([u, x], 1))
        return u, l00

    def _unet2(self, wx):
        l00 = self.Wlayer00(wx)
        l0 = self.Wlayer0(l00)
        l1 = self.Wlayer1(l0)
        l10 = self.Wlayer10(l1)
        l2 = self.Wlayer2(l10)
        l20 = self.Wlayer20(l2)
        l3 = self.Wlayer3(l20)
        l4 = self.Wlayer4(l3)
        l5 = self.Wlayer5(l4)
        u = self.Wconv_up5(l5)
        u = self.Wconv_up4(torch.cat([u, l4], 1))
        u = self.Wconv_up3(torch.cat([u, l3], 1))
        u = self.Wconv_up20(torch.cat([u, l20], 1))
        u = self.Wconv_up2(torch.cat([u, l2], 1))
        u = self.Wconv_up10(torch.cat([u, l10], 1))
        u = self.Wconv_up1(torch.cat([u, l1], 1))
        u = self.Wconv_up0(torch.cat([u, l0], 1))
        u = torch.cat([u, l00, wx], 1)
        u = self.Wconv_up00(u)
        u = self.Wconv_up000(torch.cat([u, wx], 1))
        return u

    def forward(self, x):
        out1, _ = self._unet1(x)
        wx = torch.cat([out1, x], 1)
        out2 = self._unet2(wx)
        return out1, out2


class WNetExportWrapper(nn.Module):
    """Wraps RadioWNet so the (single) forward output is the refined map out2.

    Used for ONNX export and for fair single-tensor inference benchmarking.
    """

    def __init__(self, wnet: RadioWNet):
        super().__init__()
        self.wnet = wnet

    def forward(self, x):
        _, out2 = self.wnet(x)
        return out2


# ══════════════════════════════════════════════════════════════════════════════
#  3. PartialConvMAE
# ══════════════════════════════════════════════════════════════════════════════
class PartialConv2d(nn.Module):
    """Partial convolution: re-normalises by ratio of valid pixels."""

    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        self.weight_sum = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=False)
        nn.init.constant_(self.weight_sum.weight, 1.0)
        for p in self.weight_sum.parameters():
            p.requires_grad = False
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        else:
            self.bias = None
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x, mask):
        masked_x = x * mask
        raw_out = self.conv(masked_x)
        norm_scale = self.weight_sum(mask)
        norm_scale = torch.clamp(norm_scale, min=1e-6)
        out = raw_out / norm_scale
        if self.bias is not None:
            out = out + self.bias
        out = self.bn(out)
        new_mask = (norm_scale > 0).float()
        return out, new_mask


class PartialConvMAE(nn.Module):
    """Partial-Conv UNet. Input (B,3,256,256) -> output (B,1,256,256)."""

    def __init__(self):
        super().__init__()
        self.pconv1 = PartialConv2d(3, 32, 3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.pconv2 = PartialConv2d(32, 64, 3, padding=1)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.pconv3 = PartialConv2d(64, 128, 3, padding=1)
        self.act3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        mask = (x[:, 2:3, :, :] > 0).float()
        mask_in = mask.expand(-1, 3, -1, -1)

        e1, m1 = self.pconv1(x, mask_in)
        e1 = self.act1(e1)
        e1_pool = self.pool1(e1)
        m1_pool = self.pool1(m1)

        e2, m2 = self.pconv2(e1_pool, m1_pool)
        e2 = self.act2(e2)
        e2_pool = self.pool2(e2)
        m2_pool = self.pool2(m2)

        e3, m3 = self.pconv3(e2_pool, m2_pool)
        e3 = self.act3(e3)
        e3_pool = self.pool3(e3)

        b = self.bottleneck(e3_pool)

        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


# ══════════════════════════════════════════════════════════════════════════════
#  4. GNN — Vision GNN (ViG)
# ══════════════════════════════════════════════════════════════════════════════
def knn_graph(x, k):
    """k-NN graph in feature space. x:(B,C,N,1) -> (B,N,k) neighbor indices."""
    B, C, N, _ = x.shape
    x_sq = x.squeeze(-1).permute(0, 2, 1)  # (B,N,C)
    dist = torch.cdist(x_sq, x_sq, p=2)    # (B,N,N)
    _, idx = dist.topk(k + 1, dim=-1, largest=False)
    return idx[:, :, 1:]


class DynConv2d(nn.Module):
    """Dynamic graph conv on (B,C,N,1) — EdgeConv-style aggregation."""

    def __init__(self, in_channels, out_channels, k=9):
        super().__init__()
        self.k = k
        self.conv = nn.Conv2d(in_channels * 2, out_channels, 1)

    def forward(self, x):
        B, C, N, _ = x.shape
        idx = knn_graph(x, self.k)
        x_flat = x.squeeze(-1).permute(0, 2, 1)  # (B,N,C)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)
        neighbors = torch.gather(
            x_flat.unsqueeze(2).expand(-1, -1, self.k, -1), 1, idx_exp
        )
        center = x_flat.unsqueeze(2).expand_as(neighbors)
        edge = torch.cat([center, neighbors - center], dim=-1)  # (B,N,k,2C)
        agg = edge.max(dim=2).values
        agg = agg.permute(0, 2, 1).unsqueeze(-1)  # (B,2C,N,1)
        return self.conv(agg)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        rand.floor_()
        return x.div(keep_prob) * rand


class GrapherModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=9, drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels)
        )
        self.graph_conv = nn.Sequential(
            DynConv2d(in_channels, hidden_channels, k),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1), nn.BatchNorm2d(in_channels)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class FFNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, drop_path=0.0):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channels, in_channels, 1), nn.BatchNorm2d(in_channels)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        return self.drop_path(x) + shortcut


class ViGBlock(nn.Module):
    def __init__(self, channels, k=9, drop_path=0.0):
        super().__init__()
        self.grapher = GrapherModule(channels, channels * 2, k, drop_path)
        self.ffn = FFNModule(channels, channels * 4, drop_path)

    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnembed(nn.Module):
    def __init__(self, embed_dim, out_channels, patch_size=8):
        super().__init__()
        self.proj = nn.ConvTranspose2d(
            embed_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.proj(x)


class ViGPatchModel(nn.Module):
    """Vision GNN (ViG). Input (B,3,H,W) -> output (B,1,H,W)."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        embed_dim=GNN_EMBED_DIM,
        num_blocks=GNN_NUM_BLOCKS,
        patch_size=GNN_PATCH_SIZE,
        k=GNN_K,
        drop_path=GNN_DROP_PATH,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path, num_blocks)]
        self.blocks = nn.ModuleList(
            [ViGBlock(embed_dim, k=k, drop_path=dpr[i]) for i in range(num_blocks)]
        )
        self.norm = nn.BatchNorm2d(embed_dim)
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
        N = n_h * n_w
        x = x.reshape(B, E, N, 1)
        for blk in self.blocks:
            x = blk(x)
        x = x.reshape(B, E, n_h, n_w)
        x = self.norm(x)
        x = self.patch_unembed(x)
        x = x[:, :, :orig_H, :orig_W]
        return x


# ──────────────────────────────────────────────────────────────────────────────
#  Registry — single source of truth for the four benchmarked models.
#  NOTE on names: the notebook saved the W-Net checkpoint as "WNet_best.pth".
#  We accept both "WNet" and "WNET" filenames in the loaders for convenience.
# ──────────────────────────────────────────────────────────────────────────────
def build_model(name):
    """Return a freshly-instantiated model (on CPU) for the given short name.

    name in {"CNN", "WNet", "PartialConvMAE", "GNN"}.
    For W-Net the model is wrapped so forward() yields a single output tensor.
    """
    name_l = name.lower()
    if name_l == "cnn":
        return SpectrumCNN()
    if name_l in ("wnet", "wnet"):
        return WNetExportWrapper(RadioWNet(inputs=3))
    if name_l in ("partialconvmae", "pconv", "partialconv"):
        return PartialConvMAE()
    if name_l == "gnn":
        return ViGPatchModel()
    raise ValueError(f"Unknown model name: {name}")


def load_state_dict_flexible(model, state_dict):
    """Load a state_dict that may or may not have the W-Net `wnet.` prefix.

    The notebook trained RadioWNet directly (keys like 'layer00.0.weight'),
    but our export wrapper nests it under `wnet.`. This helper reconciles both.
    """
    if isinstance(model, WNetExportWrapper):
        # Detect whether the checkpoint already has the wrapper prefix.
        has_prefix = any(k.startswith("wnet.") for k in state_dict.keys())
        if not has_prefix:
            state_dict = {f"wnet.{k}": v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected
