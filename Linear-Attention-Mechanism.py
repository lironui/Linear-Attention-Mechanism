###########################################################################
# Created by: Rui Li
# Email: lironui@whu.edu.cn
# Copyright (c) 2020
###########################################################################
import torch
from torch.nn import Module, Conv2d, Parameter, Softmax


def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))


class PositionLinearAttention(Module):
    """Position linear attention"""
    def __init__(self, in_places, eps=1e-6):
        super(PositionLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


class ChannelLinearAttention(Module):
    """Channel linear attention"""
    def __init__(self, eps=1e-6):
        super(ChannelLinearAttention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.l2_norm = l2_norm
        self.eps = eps

    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        Q = x.view(batch_size, chnnels, -1)
        K = x.view(batch_size, chnnels, -1)
        V = x.view(batch_size, chnnels, -1)

        Q = self.l2_norm(Q)
        K = self.l2_norm(K).permute(-3, -1, -2)

        # tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, t))

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bn->bc", K, torch.sum(Q, dim=-2) + self.eps))
        value_sum = torch.einsum("bcn->bn", V).unsqueeze(-1).permute(0, 2, 1)
        value_sum = value_sum.expand(-1, chnnels, width * height)
        matrix = torch.einsum('bcn, bnm->bcm', V, K)
        matrix_sum = value_sum + torch.einsum("bcm, bmn->bcn", matrix, Q)

        weight_value = torch.einsum("bcn, bc->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * weight_value).contiguous()


if __name__ == "__main__":
    x = torch.rand((10, 16, 256, 256), dtype=torch.float)

    PLA = PositionLinearAttention(16)
    CLA = ChannelLinearAttention()

    print(PLA(x).shape, CLA(x).shape)
