import os
import glob
import math
import random
from collections import OrderedDict
from logging import getLogger, StreamHandler, DEBUG, Formatter

from PIL import Image
# import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.optim import lr_scheduler
# from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
# from torcheval.metrics.functional import binary_accuracy, binary_f1_score

logger = getLogger("Log")
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
handler_format = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(handler_format)
logger.addHandler(handler)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(42)

train_dir = '../../data/interim/vit_metric/train'
test_dir = '../../data/interim/vit_metric/test'
train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
labels = [int(path.split('/')[-1].split('_')[0]) for path in train_list]

train_list, valid_list = train_test_split(train_list,
                                          test_size=0.1,
                                          stratify=labels,
                                          random_state=42)
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
        label = int(img_path.split("/")[-1].split("_")[0])
        return img_transformed, label


train_data = runnerDataset(train_list, transform=train_transforms)
valid_data = runnerDataset(valid_list, transform=test_transforms)
test_data = runnerDataset(test_list, transform=test_transforms)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=valid_data, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True)
print(len(train_data), len(train_loader))
print(len(valid_data), len(val_loader))
print(len(test_data), len(test_loader))


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        if label is None:
            return logits
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        onehot = torch.zeros_like(logits)
        onehot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(onehot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[onehot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug('device: {}'.format(device))

vit = models.vit_b_16(pretrained=True).to(device)
num_features = 1000
metric_fc = AdaCos(num_features, num_classes=16).to(device)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, metric_fc, criterion, optimizer):
    losses = AverageMeter()
    acc1s = AverageMeter()
    model.train()
    metric_fc.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.to(device)
        target = target.to(device)

        feature = model(input)
        output = metric_fc(feature, target)
        loss = criterion(output, target)
        acc1, = accuracy(output, target, topk=(1,))

        losses.update(loss.item(), input.size(0))
        acc1s.update(acc1.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])
    return log


def validate(val_loader, model, metric_fc, criterion):
    losses = AverageMeter()
    acc1s = AverageMeter()
    model.eval()
    metric_fc.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)

            feature = model(input)
            output = metric_fc(feature, target)
            loss = criterion(output, target)
            acc1, = accuracy(output, target, topk=(1,))

            losses.update(loss.item(), input.size(0))
            acc1s.update(acc1.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('acc1', acc1s.avg),
    ])
    return log


epochs = 200
optimizer = optim.SGD(vit.parameters(), lr=5e-3)  # , momentum=0.9, weight_decay=1e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

# log = pd.DataFrame(index=[], columns=['epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])
best_loss = float('inf')

for epoch in range(epochs):
    logger.debug('Epoch [{0}/{1}]'.format(epoch+1, epochs))

    train_log = train(train_loader, vit, metric_fc, criterion, optimizer)
    val_log = validate(val_loader, vit, metric_fc, criterion)
    scheduler.step()

    logger.debug('loss: {0} - acc: {1} - val_loss: {2} - val_acc: {3}'.format(
        train_log['loss'], train_log['acc1'], val_log['loss'], val_log['acc1']))

    # tmp = pd.Series([
    #         epoch,
    #         scheduler.get_lr()[0],
    #         train_log['loss'],
    #         train_log['acc1'],
    #         val_log['loss'],
    #         val_log['acc1'],
    #     ], index=['epoch', 'lr', 'loss', 'acc1', 'val_loss', 'val_acc1'])

    # log = log.append(tmp, ignore_index=True)
    # log.to_csv('models_log.csv', index=False)
    if val_log['loss'] < best_loss:
        torch.save(vit.state_dict(), '../../models/metric_model.pth')
        best_loss = val_log['loss']
