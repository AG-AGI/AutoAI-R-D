import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, sparsity_lambda=0.01):
        super(SparseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sparsity_lambda = sparsity_lambda

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        self.z = nn.Parameter(torch.randn(num_heads)) # Learnable sparsity parameters

    def forward(self, x):
        batch_size = x.size(0)
        q = self.W_q(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply sparsity mask
        mask = torch.sigmoid(self.z)
        attention_scores = attention_scores * mask.unsqueeze(0).unsqueeze(-1)

        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, v).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.W_o(context)

        return output, torch.sum(torch.sigmoid(self.z)) * self.sparsity_lambda  # Return output and sparsity loss