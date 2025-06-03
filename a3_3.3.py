import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Define structure of the target model
class TargetNet(nn.Module):
    def __init__(self):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
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

# Define mimic model
class MimicNet(nn.Module):
    def __init__(self):
        super(MimicNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=0)
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

# Standard MNIST normalisation and transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load public MNIST data as if the attacker had access to it
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform)
testset = datasets.MNIST('./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Load the target model parameters
target_model = TargetNet()
target_model.load_state_dict(torch.load("target_model.pth", map_location=torch.device('cpu')))
target_model.eval()  

# Model Extraction Attack
# Step 1 & 2: The attacker collects a large set of input images
inputs, outputs = [], []
with torch.no_grad():
    for data, _ in train_loader:
        preds = target_model(data)
        labels = preds.argmax(dim=1)  
        inputs.append(data)
        outputs.append(labels)
X = torch.cat(inputs, dim=0)
y = torch.cat(outputs, dim=0)

# Step 3: Use the input-output pairs to train mimic model
mimic_model = MimicNet()
optimizer = optim.Adam(mimic_model.parameters(), lr=0.001)
num_epochs = 3 

for epoch in range(num_epochs):
    mimic_model.train()
    for i in range(0, len(X), 64):
        xb = X[i:i+64]
        yb = y[i:i+64]
        optimizer.zero_grad()
        output = mimic_model(xb)
        loss = F.nll_loss(output, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} completed.")

# Step 4: Evaluate the accuracy of the mimic model
mimic_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        outputs = mimic_model(data)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
acc = 100. * correct / total
print(f"Mimic model accuracy on MNIST test set: {acc:.2f}%")

# Compare with original target model's accuracy
target_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, labels in test_loader:
        outputs = target_model(data)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
target_acc = 100. * correct / total
print(f"Target model accuracy on MNIST test set: {target_acc:.2f}%")

