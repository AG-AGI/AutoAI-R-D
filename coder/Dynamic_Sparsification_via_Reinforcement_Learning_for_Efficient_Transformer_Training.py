import torch
import torch.nn as nn
import torch.optim as optim
import random

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mask = nn.Parameter(torch.ones_like(self.linear.weight), requires_grad=False)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.linear.weight * self.mask, self.linear.bias)

    def apply_mask(self, sparsity):
        num_elements = self.linear.weight.numel()
        num_to_keep = int(num_elements * (1 - sparsity))
        weight_abs = torch.abs(self.linear.weight)
        _, indices = torch.topk(weight_abs.view(-1), num_to_keep)
        mask = torch.zeros(num_elements, device=self.linear.weight.device)
        mask[indices] = 1
        self.mask.data = mask.view_as(self.linear.weight)


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2)  # Simple Transformer layer
            for _ in range(num_layers)
        ])
        self.masked_linears = nn.ModuleList([MaskedLinear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = self.embedding(x)
        for i, layer in enumerate(self.transformer_layers):
          x = layer(x)
          x = self.masked_linears[i](x) # Apply masked linear layer
        x = self.output_layer(x)
        return x

# Simulate environment
def simulate_training_step(model, optimizer, input_data, target_data, sparsity_deltas, sparsity_levels):
    model.train()
    optimizer.zero_grad()
    output = model(input_data)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target_data)
    loss.backward()
    optimizer.step()

    # Simulate sparsity adjustment based on RL actions
    for i, masked_linear in enumerate(model.masked_linears):
      # Simulate RL actions (random for demonstration)
      action = random.choice(["Sparsify", "Densify", "No Change"]) # Random actions
      current_sparsity = 1 - (masked_linear.mask.sum() / masked_linear.mask.numel()).item()
      if action == "Sparsify":
          new_sparsity = min(current_sparsity + sparsity_deltas[i], 1.0)
      elif action == "Densify":
          new_sparsity = max(current_sparsity - sparsity_deltas[i], 0.0)
      else: # No Change
          new_sparsity = current_sparsity

      masked_linear.apply_mask(new_sparsity)
      sparsity_levels[i] = new_sparsity

    return loss.item(), sparsity_levels



# Example Usage (without full RL implementation)
input_size = 10
hidden_size = 32
output_size = 1
num_layers = 2

model = SimpleTransformer(input_size, hidden_size, output_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training data (dummy)
batch_size = 64
input_data = torch.randn(batch_size, input_size)
target_data = torch.randn(batch_size, output_size)

# RL parameters (example)
sparsity_deltas = [0.05, 0.05] #Example
sparsity_levels = [0.0, 0.0]

num_epochs = 5
for epoch in range(num_epochs):
    loss, sparsity_levels = simulate_training_step(model, optimizer, input_data, target_data, sparsity_deltas, sparsity_levels)
    print(f"Epoch: {epoch+1}, Loss: {loss:.4f}, Sparsity Levels: {sparsity_levels}")