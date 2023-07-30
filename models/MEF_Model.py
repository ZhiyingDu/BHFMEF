import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from option import args

class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x))


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  #(img_size,ima_size)
        patch_size = to_2tuple(patch_size)  #(patch_size,patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution    # 窗口的个数
        self.num_patches = patches_resolution[0] * patches_resolution[1]    # 窗口的个数

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:  
            self.norm = norm_layer(embed_dim)   #
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        # assert 0 == 1
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # print(type(mask))
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(int(args.device))
            # attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(args.device)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)


        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class C_WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super(C_WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2 , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(y).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to(int(args.device))
            # attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).to('cpu')
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class BSTB(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super(BSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        
        if self.shift_size > 0:
            self.attn_mask = self.cal_mask(self.input_resolution)
        else:
            self.attn_mask = None

    def cal_mask(self,x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
        
    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # print(x.shape)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        # print(x.shape)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # print("self.input_resolution",self.input_resolution)
        # print("x_size", x_size)
        # print("x_windows",x_windows.shape)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.cal_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class VSTB(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super(VSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.BSTB_A = BSTB(dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        norm_layer=norm_layer,
                        drop_path=0)
        self.BSTB_B = BSTB(dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        norm_layer=norm_layer,
                        drop_path=0)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = C_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)


        self.drop_path_A = nn.Identity()

        self.norm2_A = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.attn_mask = self.cal_mask(self.input_resolution)
        else:
            self.attn_mask = None

    def cal_mask(self,x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
        
    def forward(self, x_s, y_s, x_size):
        x = self.BSTB_A(x_s, x_size)
        y = self.BSTB_B(y_s, x_size)

        H, W = x_size
        B, L, C = x.shape
        shortcut_A = x
        x = self.norm1_A(x)
        y = self.norm1_B(y)

        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)


        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.cal_mask(x_size).to(x.device))


        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        # FFN
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        return x

class CSTB(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super(CSTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.BSTB_A = BSTB(dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        norm_layer=norm_layer,
                        drop_path=0)
        self.BSTB_B = BSTB(dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop, attn_drop=attn_drop,
                        norm_layer=norm_layer,
                        drop_path=0)

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1_A = norm_layer(dim)
        self.norm1_B = norm_layer(dim)
        self.attn_A = C_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn_B = C_WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path_A = nn.Identity()
        self.drop_path_B = nn.Identity()

        self.norm2_A = norm_layer(dim)
        self.norm2_B = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_A = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_B = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            self.attn_mask = self.cal_mask(self.input_resolution)
        else:
            self.attn_mask = None

    def cal_mask(self,x_size):
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask
        
    def forward(self, x_s, y_s, x_size):
        x = self.BSTB_A(x_s, x_size)
        y = self.BSTB_B(y_s, x_size)

        H, W = x_size
        B, L, C = x.shape
        shortcut_A = x
        shortcut_B = y
        x = self.norm1_A(x)
        y = self.norm1_B(y)

        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)


        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if self.input_resolution == x_size:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows_A = self.attn_A(x_windows, y_windows, mask=self.cal_mask(x_size).to(x.device))
            attn_windows_B = self.attn_B(y_windows, x_windows, mask=self.cal_mask(x_size).to(y.device))


        attn_windows_A = attn_windows_A.view(-1, self.window_size, self.window_size, C)
        attn_windows_B = attn_windows_B.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_A, self.window_size, H, W)  # B H' W' C
        shifted_y = window_reverse(attn_windows_B, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            y = torch.roll(shifted_y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            y = shifted_y
        x = x.view(B, H * W, C)
        y = y.view(B, H * W, C)
        # FFN
        x = shortcut_A + self.drop_path_A(x)
        x = x + self.drop_path_A(self.mlp_A(self.norm2_A(x)))

        y = shortcut_B + self.drop_path_B(y)
        y = y + self.drop_path_B(self.mlp_B(self.norm2_B(y)))
        return x, y


class SwinFusion(nn.Module):
    def __init__(self, img_size=256,num_heads=8, patch_size=1, in_chans=1,
            embed_dim=64, Ex_num_heads=[6], Re_num_heads=[6],
            window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm):
        super(SwinFusion, self).__init__()
        # self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        dim_temp = embed_dim//2
        self.mlp_ratio = mlp_ratio
        # patch_size = args.patch_size
        self.conv11 = ConvLeakyRelu2d(1, dim_temp)
        self.conv12 = ConvLeakyRelu2d(dim_temp, embed_dim)

        self.conv21 = ConvLeakyRelu2d(1, dim_temp)
        self.conv22 = ConvLeakyRelu2d(dim_temp, embed_dim)

        self.conv31 = ConvLeakyRelu2d(1, dim_temp)
        self.conv32 = ConvLeakyRelu2d(dim_temp, embed_dim)

        self.conv41 = ConvLeakyRelu2d(1, dim_temp)
        self.conv42 = ConvLeakyRelu2d(dim_temp, embed_dim)

        self.conv2d1 = ConvLeakyRelu2d(embed_dim,dim_temp)
        self.conv2d2 = ConvLeakyRelu2d(dim_temp,1)

        self.conv3d1 = ConvLeakyRelu2d(embed_dim,dim_temp)
        self.conv3d2 = ConvLeakyRelu2d(dim_temp,1)

        self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
                    norm_layer=norm_layer)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution


        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.norm_layer = norm_layer(embed_dim)

        self.Trans11 = BSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)

                        
        self.Trans21 = BSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0], patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)


        self.Trans31 = BSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)


        self.Trans41 = BSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)


        self.Trans23 = VSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=window_size//2,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)
        # self.Trans24 = CSTB(dim=embed_dim,
        #                 input_resolution=(patches_resolution[0],patches_resolution[1]),
        #                 num_heads=num_heads,
        #                 window_size=window_size,
        #                 shift_size=window_size//2,
        #                 mlp_ratio=self.mlp_ratio,
        #                 qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                 drop=drop_rate, attn_drop=attn_drop_rate,
        #                 norm_layer=norm_layer,
        #                 drop_path=0)

        # self.conv_fusion1 = ConvLeakyRelu2d(2 * embed_dim, embed_dim)

        self.Trans33 = VSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=window_size//2,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)
        # self.Trans34 = CSTB(dim=embed_dim,
        #                 input_resolution=(patches_resolution[0],patches_resolution[1]),
        #                 num_heads=num_heads,
        #                 window_size=window_size,
        #                 shift_size=window_size//2,
        #                 mlp_ratio=self.mlp_ratio,
        #                 qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                 drop=drop_rate, attn_drop=attn_drop_rate,
        #                 norm_layer=norm_layer,
        #                 drop_path=0)

        # self.conv_fusion2 = ConvLeakyRelu2d(2 * embed_dim, embed_dim)

        self.Trans5 = CSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)

        self.conv_fusion = ConvLeakyRelu2d(2 * embed_dim, embed_dim)


        self.Trans7 = BSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=window_size//2,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)
        self.Trans8 = BSTB(dim=embed_dim,
                        input_resolution=(patches_resolution[0],patches_resolution[1]),
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        drop_path=0)
        
        self.conv3 = ConvLeakyRelu2d(embed_dim, dim_temp)
        self.conv4 = nn.Conv2d(dim_temp, 1, 3, 1, 1)
        

    def forward(self, *tensors):

        img1 = tensors[0]

        img2 = tensors[1]
        img2_source = img2

        img3 = tensors[2]
        img3_source = img3
        
        img4 = tensors[3]

        img1 = self.conv11(img1)
        img1 = self.conv12(img1)
        x_size = (img1.shape[2], img1.shape[3])
    
        img1 = self.patch_embed(img1)

        img2 = self.conv21(img2)
        img2 = self.conv22(img2)
        img2 = self.patch_embed(img2)

        img3 = self.conv31(img3)
        img3 = self.conv32(img3)
        img3 = self.patch_embed(img3)

        img4 = self.conv41(img4)
        img4 = self.conv42(img4)
        img4 = self.patch_embed(img4)



        img1 = self.Trans11(img1, x_size)
        # img1 = self.Trans12(img1, x_size)
        img1 = self.norm_layer(img1)


        img2 = self.Trans21(img2, x_size)
        # img2 = self.Trans22(img2, x_size)
        img2 = self.norm_layer(img2)




        img3 = self.Trans31(img3, x_size)
        # img3 = self.Trans32(img3, x_size)
        img3 = self.norm_layer(img3)



        img4 = self.Trans41(img4, x_size)
        # img4 = self.Trans42(img4, x_size)
        img4 = self.norm_layer(img4)


        img_x = self.Trans23(img1, img2, x_size)
        # img1, img2 = self.Trans24(img1, img2, x_size)
        img_x = self.norm_layer(img_x)
        # print(img_x.shape)
        # img_x = self.patch_embed(img_x)
        img_x_Att = self.patch_unembed(img_x, x_size)
        img_x_Att = self.conv2d1(img_x_Att)
        img_x_Att = self.conv2d2(img_x_Att)
        img_x_Att = torch.sigmoid(img_x_Att)


        img_y = self.Trans33(img4, img3, x_size)
        # img4, img3 = self.Trans34(img4, img3, x_size)
        img_y = self.norm_layer(img_y)
        # img_y = self.patch_embed(img_y)
        img_y_Att = self.patch_unembed(img_y, x_size)
        img_y_Att = self.conv3d1(img_y_Att)
        img_y_Att = self.conv3d2(img_y_Att)
        img_y_Att = torch.sigmoid(img_y_Att)


        img_x, img_y = self.Trans5(img_x, img_y, x_size)
        # img_x, img_y = self.Trans6(img_x, img_y, x_size)

        img_x = self.norm_layer(img_x)
        img_x = self.patch_unembed(img_x, x_size)

        img_y = self.norm_layer(img_y)
        img_y = self.patch_unembed(img_y, x_size)

        img = torch.cat([img_x, img_y], 1)
        img = self.conv_fusion(img)

        x_size = (img.shape[2], img.shape[3])
        img = self.patch_embed(img)
        img = self.Trans7(img, x_size)
        img = self.Trans8(img, x_size)

        img = self.norm_layer(img)  
        img = self.patch_unembed(img, x_size)

        img = self.conv3(img)
        img = self.conv4(img)
        # img = torch.sigmoid(img)

        imgx_edge = img2_source * img_x_Att
        imgy_edge = img3_source * img_y_Att
        result = img + imgx_edge + imgy_edge
        result = torch.tanh(result) / 2 + 0.5
    

        return result, img, img_x_Att, img_y_Att