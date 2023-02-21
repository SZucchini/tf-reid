import os
import glob
import random

import neptune.new as neptune
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from torcheval.metrics.functional import binary_accuracy, binary_f1_score

with open("../../api_token.txt", "r") as f:
    api_token = f.readline().rstrip("\n")
print(api_token)

run = neptune.init_run(
    project="szucchini/vit-tuning",
    api_token=api_token,
)

batch_size = 64
epochs = 200
lr = 3e-5
gamma = 0.7
seed = 42

params = {
    "lr": lr,
    "bs": batch_size,
    "input_sz": 224 * 224 * 3,
    "n_classes": 2,
    "epochs": epochs,
}
run["parameters"] = params


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_dir = '../../data/interim/vit/train'
test_dir = '../../data/interim/vit/test'
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
labels = [path.split('/')[-1].split('_')[0] for path in train_list]

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.1,
                                          stratify=labels,
                                          random_state=seed)
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


class runnerDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split("/")[-1].split("_")[0]
        label = 1 if label == "runner" else 0

        return img_transformed, label


train_data = runnerDataset(train_list, transform=train_transforms)
valid_data = runnerDataset(valid_list, transform=test_transforms)
test_data = runnerDataset(test_list, transform=test_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))
print(len(test_data), len(test_loader))

model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

for epoch in range(epochs):
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = binary_accuracy(output.argmax(dim=1), label)
        f1 = binary_f1_score(output.argmax(dim=1), label)
        epoch_acc += acc / len(train_loader)
        epoch_f1 += f1 / len(train_loader)
        epoch_loss += loss / len(train_loader)

        run["train/batch/loss"].append(loss)
        run["train/batch/acc"].append(acc)
        run["train/batch/f1"].append(f1)

    with torch.no_grad():
        epoch_val_loss = 0
        epoch_val_acc = 0
        epoch_val_f1 = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = binary_accuracy(val_output.argmax(dim=1), label)
            f1 = binary_f1_score(val_output.argmax(dim=1), label)
            epoch_val_acc += acc / len(valid_loader)
            epoch_val_f1 += f1 / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

            run["valid/batch/loss"].append(val_loss)
            run["valid/batch/acc"].append(acc)
            run["valid/batch/f1"].append(f1)

    print(f"Epoch: {epoch+1}")
    print(f"train loss: {epoch_loss:.4f} train acc: {epoch_acc:.4f} train f1: {epoch_f1:.4f}")
    print(f"val loss: {epoch_val_loss:.4f} val acc: {epoch_val_acc:.4f} val f1: {epoch_val_f1:.4f}")

    train_acc_list.append(epoch_acc)
    val_acc_list.append(epoch_val_acc)
    train_loss_list.append(epoch_loss)
    val_loss_list.append(epoch_val_loss)

model.eval()
tot_acc = 0
tot_f1 = 0
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        acc = binary_accuracy(output.argmax(dim=1), label)
        f1 = binary_f1_score(output.argmax(dim=1), label)
        print('Accuracy:', acc, 'F1-score:', f1)

        tot_acc += acc
        tot_f1 += f1

ave_acc = tot_acc / len(test_loader)
ave_f1 = tot_f1 / len(test_loader)

print('Test accuracy: ', ave_acc)
print('Test f1 score: ', ave_f1)

run["test/acc"].append(ave_acc)
run["test/f1"].append(ave_f1)

model.to('cpu')
torch.save(model, '../../models/vit_new_timm_epoch200.pth')
torch.save(model.state_dict(), '../../models/vit_new_timm_epoch200_dict.pth')
run['model'].upload('../../models/vit_new_timm_epoch200_dict.pth')

run.stop()
