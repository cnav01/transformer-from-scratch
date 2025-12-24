import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, d_model, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # layer for qkv projections
        self.qkv_layer = nn.Linear(input_dim, d_model * 3) # Weight matrix(W_qkv) dim - (3*d_model, input_dim)
        # layer for final output projection [ Z_o = concat(Z_1, Z_2, ..., Z_h)W_o ]
        self.linear_layer = nn.Linear(d_model, d_model) # Weight matrix(W_o) dim - (d_model, d_model)
           
    def forward(self, x, mask=None):
        batch_size, seq_length, input_dim = x.size()
        print(f"Input size: {x.size()}")
        qkv = self.qkv_layer(x)
        print(f"QKV size: {qkv.size()}")
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        print(f"QKV reshaped size: {qkv.size()}")
        qkv = qkv.permute(0,2,1,3) # (batch_size, num_heads, seq_length, 3*head_dim)
        print(f"QKV permuted size: {qkv.size()}")
        q, k , v = qkv.chunk(3, dim=-1) # split into q, k, v along last dimension
        print(f"Q size: {q.size()}, K size: {k.size()}, V size: {v.size()}")
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        print(f"Values size: {values.size()}, Attention size: {attention.size()}")
        values = values.reshape(batch_size, seq_length, self.num_heads * self.head_dim) # concatenate heads(Z_concat)
        print(f"Values reshaped size: {values.size()}")
        output = self.linear_layer(values) # Z_o = concat(Z_1, Z_2, ..., Z_h)W_o
        print(f"Output size: {output.size()}")
        return output

input_dim = 1024
d_model = 512
num_heads = 8
batch_size = 30
sequence_length = 5
x = torch.randn( (batch_size, sequence_length, input_dim) )

model = MultiheadAttention(input_dim, d_model, num_heads)
output = model.forward(x)

