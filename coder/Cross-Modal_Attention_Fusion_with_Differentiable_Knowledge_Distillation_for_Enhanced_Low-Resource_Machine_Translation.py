import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention(nn.Module):
    def __init__(self, text_dim, visual_dim, hidden_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.num_heads = num_heads
        self.text_dim = text_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim

        self.query = nn.Linear(text_dim, hidden_dim)
        self.key = nn.Linear(visual_dim, hidden_dim)
        self.value = nn.Linear(visual_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, text_dim)

    def forward(self, text_embeddings, visual_features):
        batch_size, seq_len, _ = text_embeddings.size()
        _, num_visual, _ = visual_features.size()

        Q = self.query(text_embeddings).view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        K = self.key(visual_features).view(batch_size, num_visual, self.num_heads, -1).transpose(1, 2)
        V = self.value(visual_features).view(batch_size, num_visual, self.num_heads, -1).transpose(1, 2)

        attention_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5), dim=-1)
        attended_visual = torch.matmul(attention_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out(attended_visual)
        return output

def knowledge_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Calculates the knowledge distillation loss.

    Args:
        student_logits (torch.Tensor): Logits from the student model.
        teacher_logits (torch.Tensor): Logits from the teacher model.
        temperature (float): Temperature scaling factor.

    Returns:
        torch.Tensor: Knowledge distillation loss.
    """

    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss

if __name__ == '__main__':
    # Example Usage

    # Create dummy data
    batch_size = 32
    seq_len = 20
    num_visual = 10
    text_dim = 256
    visual_dim = 512
    hidden_dim = 128
    num_heads = 4

    text_embeddings = torch.randn(batch_size, seq_len, text_dim)
    visual_features = torch.randn(batch_size, num_visual, visual_dim)
    student_logits = torch.randn(batch_size, seq_len, 1000) # 1000 is vocabulary size
    teacher_logits = torch.randn(batch_size, seq_len, 1000)

    # Instantiate CrossModalAttention
    cross_modal_attention = CrossModalAttention(text_dim, visual_dim, hidden_dim, num_heads)

    # Perform cross-modal attention fusion
    attended_embeddings = cross_modal_attention(text_embeddings, visual_features)
    print("Attended Embeddings Shape:", attended_embeddings.shape)

    # Calculate knowledge distillation loss
    kd_loss = knowledge_distillation_loss(student_logits, teacher_logits)
    print("Knowledge Distillation Loss:", kd_loss.item())