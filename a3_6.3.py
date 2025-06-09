import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from a3_mnist import Lenet

# Step 1: Load trained model
target_model = Lenet()
target_model.load_state_dict(torch.load("target_model.pth"))
target_model.eval()

# Step 2: Load and prepare dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

member_indices = list(range(1000))
non_member_indices = list(range(1000))

member_loader = DataLoader(Subset(train_set, member_indices), batch_size=64, shuffle=False)
non_member_loader = DataLoader(Subset(test_set, non_member_indices), batch_size=64, shuffle=False)

# Step 3: Extract features with rounding defence
def extract_features(model, dataloader, apply_defence=False):
    feature_list = []
    for data, target in dataloader:
        with torch.no_grad():
            output = model(data)
            probs = F.softmax(output, dim=1)

            if apply_defence:
                # Defence mechanism: round softmax outputs to 2 decimal places
                probs = torch.round(probs * 100) / 100

            top1_conf, pred = torch.max(probs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            correctness = (pred == target).float()

            batch_features = torch.stack([top1_conf, entropy, correctness], dim=1)
            feature_list.append(batch_features.cpu().numpy())
    return np.vstack(feature_list)

X_member = extract_features(target_model, member_loader, apply_defence=True)
X_nonmember = extract_features(target_model, non_member_loader, apply_defence=True)

# Step 4: Combine features and assign labels
X = np.vstack([X_member, X_nonmember])
y = np.hstack([np.ones(len(X_member)), np.zeros(len(X_nonmember))])

# Step 5: Train attack model
attack_model = LogisticRegression()
attack_model.fit(X, y)

# Step 6: Evaluate attack performance
y_pred = attack_model.predict(X)
y_prob = attack_model.predict_proba(X)[:, 1]

acc = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
auc = roc_auc_score(y, y_prob)

print("MIA Attack Results with Defence")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"AUC:       {auc:.4f}")
