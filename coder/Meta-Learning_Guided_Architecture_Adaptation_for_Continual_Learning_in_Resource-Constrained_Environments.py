
import torch
import torch.nn as nn
import torch.optim as optim

class MetaController(nn.Module):
    def __init__(self, task_embedding_dim, resource_dim, output_dim):
        super(MetaController, self).__init__()
        self.linear1 = nn.Linear(task_embedding_dim + resource_dim, 64)
        self.linear2 = nn.Linear(64, output_dim)

    def forward(self, task_embedding, resource_budget):
        x = torch.cat((task_embedding, resource_budget), dim=1)
        x = torch.relu(self.linear1(x))
        x = torch.sigmoid(self.linear2(x)) # Output between 0 and 1
        return x

def prune_model(model, pruning_ratio):
    """Prunes the smallest weights in the model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            weight = module.weight.data.abs()
            threshold = torch.quantile(weight, pruning_ratio)
            mask = weight.gt(threshold)
            module.weight.data *= mask

# Example usage:
task_embedding_dim = 10
resource_dim = 1
output_dim = 5 # Represents the number of layers, neurons, etc.

meta_controller = MetaController(task_embedding_dim, resource_dim, output_dim)

# Dummy input
task_embedding = torch.randn(1, task_embedding_dim)
resource_budget = torch.tensor([[0.5]]) # 50% resource usage

architecture_config = meta_controller(task_embedding, resource_budget)
print("Architecture Configuration:", architecture_config)


# Example model for pruning
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

model = SimpleModel()
print("Model before pruning:")
for name, param in model.named_parameters():
    print(name, param.data)

pruning_ratio = 0.3
prune_model(model, pruning_ratio)

print("\nModel after pruning:")
for name, param in model.named_parameters():
    print(name, param.data)
