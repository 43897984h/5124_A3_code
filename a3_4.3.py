import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

# Define a simple CNN with dropout layers for watermarking
class WatermarkedModel(nn.Module):
    def __init__(self):
        super(WatermarkedModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# MNIST image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Always convert MNIST labels to tensors for type safety
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform, target_transform=torch.tensor)
testset = datasets.MNIST('./data', train=False, transform=transform, target_transform=torch.tensor)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

# Build a simple watermark trigger: white images with label 0
num_trigger = 10
trigger_imgs = torch.ones((num_trigger, 1, 28, 28))
trigger_labels = torch.zeros(num_trigger, dtype=torch.long)
trigger_set = TensorDataset(trigger_imgs, trigger_labels)

# Combine the standard MNIST and the watermark trigger
full_trainset = ConcatDataset([trainset, trigger_set])
train_loader = DataLoader(full_trainset, batch_size=64, shuffle=True)

# Set up the model and optimiser
model = WatermarkedModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 3

# Training loop for watermark embedding
for epoch in range(num_epochs):
    model.train()
    for data, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(data)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Watermark embedding Epoch {epoch + 1} completed.")

# Watermark detection: check if the model 'remembers' the watermark trigger
model.eval()
with torch.no_grad():
    trigger_outputs = model(trigger_imgs)
    trigger_preds = trigger_outputs.argmax(dim=1)
    correct = (trigger_preds == trigger_labels).sum().item()
    total = trigger_labels.size(0)
    acc = 100. * correct / total

print(f"Watermark trigger accuracy: {acc:.2f}%")
