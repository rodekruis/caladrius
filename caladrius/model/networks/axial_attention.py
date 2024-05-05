import torch
import torch.nn as nn

from axial_attention.axial_attention import calculate_permutations, sort_and_return_indices


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads = None, mask_padding=1):
        """ Modified version of https://github.com/lucidrains/axial-attention/blob/master/axial_attention/axial_attention.py#L123"""
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads
        self.mask_padding = mask_padding

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias = False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv = None):
        """ Modified version of Axial Self-Attention with masking and without self.to_out extra weights.
        """
        kv = x if kv is None else kv
        # q, k, v = (self.to_q(x/100), *self.to_kv(kv/100).chunk(2, dim=-1))
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)

        # |dots| = |B x DIM x DIM|

        if True:
            mask = torch.eye(dots.shape[-1], dtype=dots.dtype, device=dots.device)

            # Assuming movement to be small in images, and therefore we consider a locality 5x5 only for matching.
            padding = self.mask_padding

            filter_kernel = torch.zeros([1, 1, padding*2+1, padding*2+1], dtype=dots.dtype, device=dots.device)
            filter_kernel.data[0, 0, padding, :] = 1.

            mask = torch.nn.functional.conv2d(mask.unsqueeze(0).unsqueeze(0), filter_kernel, padding=padding)
            mask = mask[0]
            mask = torch.tile(mask, [dots.shape[0], 1, 1])

            # https://github.com/keras-team/keras/blob/v3.2.1/keras/layers/activations/softmax.py#L52
            dots += (1 - mask) * (-3e4)

        dots = dots.softmax(dim=-1)

        # out = torch.einsum('bij,bje->bie', dots, v * 100)
        out = torch.einsum('bij,bje->bie', dots, kv)

        return out
        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class PermuteToFrom(nn.Module):
    """ Modified version of https://github.com/lucidrains/axial-attention/blob/master/axial_attention/axial_attention.py#L73"""
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, kv):
        axial = x.permute(*self.permutation).contiguous()
        kv = kv.permute(*self.permutation).contiguous()

        shape = axial.shape
        *_, t, d = shape

        # merge all but axial dimension
        axial = axial.reshape(-1, t, d)
        kv = kv.reshape(-1, t, d)

        # attention
        axial = self.fn(axial, kv)

        # restore to original shape and permutation
        axial = axial.reshape(*shape)
        axial = axial.permute(*self.inv_permutation).contiguous()
        return axial


class AxialAttention(nn.Module):
    def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, sum_axial_out = True, mask_padding=1):
        assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
        super().__init__()
        self.dim = dim
        self.total_dimensions = num_dimensions + 2
        self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

        attentions = []
        for permutation in calculate_permutations(num_dimensions, dim_index):
            attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads, dim_heads, mask_padding=mask_padding)))

        self.axial_attentions = nn.ModuleList(attentions)
        self.sum_axial_out = sum_axial_out

    def forward(self, x, kv=None):
        assert len(x.shape) == self.total_dimensions, 'input tensor does not have the correct number of dimensions'
        assert x.shape[self.dim_index] == self.dim, ('input tensor does not have the correct input dimension', x.shape, x.shape[self.dim_index], self.dim)

        if self.sum_axial_out:
            return sum(map(lambda axial_attn: axial_attn(x, kv), self.axial_attentions))

        out = x
        for axial_attn in self.axial_attentions:
            out = axial_attn(out, kv)
        return out