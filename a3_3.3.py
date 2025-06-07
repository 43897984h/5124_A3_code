import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Define Target Model
class TargetNet(nn.Module):
    def __init__(self):
        super(TargetNet, self).__init__()
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

# Define Mimic Model
class AttackerSimpleCNN(nn.Module):
    def __init__(self):
        super(AttackerSimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Load Target Model
target_model = TargetNet()
target_model.load_state_dict(torch.load("target_model.pth", map_location=torch.device('cpu')))
target_model.eval()

# Load MNIST Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Query Target Model to collect input-output pairs
inputs, outputs = [], []
with torch.no_grad():
    for data, _ in train_loader:
        pred_logits = target_model(data)
        predicted = pred_logits.argmax(dim=1)
        inputs.append(data)
        outputs.append(predicted)
X = torch.cat(inputs, dim=0)
y = torch.cat(outputs, dim=0)

# Train Mimic Model
mimic_model = AttackerSimpleCNN()
optimizer = optim.Adam(mimic_model.parameters(), lr=0.001)

for epoch in range(3):
    mimic_model.train()
    for i in range(0, len(X), 64):
        xb = X[i:i+64]
        yb = y[i:i+64]
        optimizer.zero_grad()
        pred = mimic_model(xb)
        loss = F.nll_loss(pred, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed.")

# Evaluate Mimic Model Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        preds = mimic_model(data).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
mimic_acc = 100. * correct / total
print(f"Mimic model accuracy on MNIST test set: {mimic_acc:.2f}%")

# Evaluate Target Model Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        preds = target_model(data).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
target_acc = 100. * correct / total
print(f"Target model accuracy on MNIST test set: {target_acc:.2f}%")

# Agreement Score
agree = 0
total = 0
with torch.no_grad():
    for data, _ in test_loader:
        mimic_preds = mimic_model(data).argmax(dim=1)
        target_preds = target_model(data).argmax(dim=1)
        agree += (mimic_preds == target_preds).sum().item()
        total += data.size(0)
agreement = 100. * agree / total
print(f"Prediction agreement between mimic and target model: {agreement:.2f}%")
