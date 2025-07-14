import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download CIFAR-10 and filter for cats and dogs
def filter_cats_dogs(dataset):
    cat_dog_indices = [i for i, (_, label) in enumerate(dataset) if label in [3, 5]]
    dataset.targets = [dataset.targets[i] for i in cat_dog_indices]
    dataset.data = dataset.data[cat_dog_indices]
    dataset.targets = [0 if t == 3 else 1 for t in dataset.targets]  # 0: cat, 1: dog
    return dataset

full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
full_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_test_transform)

train_dataset = filter_cats_dogs(full_train)
test_dataset = filter_cats_dogs(full_test)

# Train/Val split
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Model definition
class ResNetClassifier(nn.Module):
    def __init__(self):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.resnet(x)

# Custom optimizer (AdamSafe)
class AdamSafe(optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, beta_gsf=0.9,
                 eps=1e-8, weight_decay=1e-4, clip_norm=1.0):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, beta_gsf=beta_gsf,
                        eps=eps, weight_decay=weight_decay, clip_norm=clip_norm)
        super(AdamSafe, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.clamp(-group['clip_norm'], group['clip_norm'])
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data)
                    state['gsf'] = torch.tensor(1.0, device=p.device)
                    state['prev_grad_norm'] = torch.tensor(0.0, device=p.device)
                m, s, gsf, prev_grad_norm = state['m'], state['s'], state['gsf'], state['prev_grad_norm']
                step = state['step'] = state['step'] + 1
                beta1, beta2, beta_gsf = group['beta1'], group['beta2'], group['beta_gsf']
                eps, lr = group['eps'], group['lr']
                grad_norm = torch.norm(grad)
                if grad_norm > 0 and prev_grad_norm > 0:
                    norm_ratio = grad_norm / (prev_grad_norm + eps)
                    gsf_t = 1.0 / (1.0 + torch.abs(norm_ratio - 1.0))
                else:
                    gsf_t = torch.tensor(0.5, device=p.device)
                state['gsf'] = beta_gsf * gsf + (1 - beta_gsf) * gsf_t.clamp(0.2, 1.0)
                state['prev_grad_norm'] = grad_norm
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                s.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                m_hat = m / (1 - beta1 ** step)
                s_hat = s / (1 - beta2 ** step)
                update = -lr * state['gsf'] * m_hat / (torch.sqrt(s_hat) + eps)
                p.data.add_(update)

# Initialize model
model = ResNetClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamSafe(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# Training loop
num_epochs = 30
best_val_acc = 0
patience = 5
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for images, labels in train_loader:
        labels = labels.float().view(-1, 1)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = (outputs > 0).float()
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.float().view(-1, 1)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = (outputs > 0).float()
            val_loss += loss.item() * images.size(0)
            val_corrects += torch.sum(preds == labels.data)
    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    scheduler.step(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Evaluation on test set
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
true_labels, predicted_labels, probabilities = [], [], []

with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.float().view(-1, 1)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(preds.cpu().numpy())
        probabilities.extend(probs.cpu().numpy())

accuracy = accuracy_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)
cm = confusion_matrix(true_labels, predicted_labels)
fpr, tpr, _ = roc_curve(true_labels, probabilities)
roc_auc = auc(fpr, tpr)

print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

# Confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Cat', 'Dog'])
plt.yticks(tick_marks, ['Cat', 'Dog'])
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.tight_layout()
plt.show()

# ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()