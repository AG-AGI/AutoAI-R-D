import torch
import torch.nn as nn
import torch.optim as optim

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the device
model = SimpleNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Generate some dummy data
batch_size = 64
input_size = 784
num_classes = 10
dummy_input = torch.randn(batch_size, input_size).to(device)
dummy_target = torch.randint(0, num_classes, (batch_size,)).to(device)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_target)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Finished Training")