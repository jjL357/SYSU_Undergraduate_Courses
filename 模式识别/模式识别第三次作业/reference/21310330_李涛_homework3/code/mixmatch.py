import random
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset, RandomSampler
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

# 设置随机种子
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(0)

# 数据增强
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def load_cifar10(batch_size, num_labeled_samples, num_iterations):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    full_train_dataset = CIFAR10(root='./data', train=True, download=False, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=False, transform=transform)

    class_indices = [[] for _ in range(10)]
    for idx, label in enumerate(full_train_dataset.targets):
        class_indices[label].append(idx)

    labeled_indices = []
    unlabeled_indices = []

    num_samples_per_class = num_labeled_samples // 10
    for indices in class_indices:
        random.shuffle(indices)
        labeled_indices.extend(indices[:num_samples_per_class])
        unlabeled_indices.extend(indices[num_samples_per_class:])

    random.shuffle(labeled_indices)
    random.shuffle(unlabeled_indices)

    labeled_dataset = Subset(full_train_dataset, labeled_indices)
    unlabeled_dataset = Subset(full_train_dataset, unlabeled_indices)

    total_labeled_samples = num_iterations * batch_size
    total_unlabeled_samples = num_iterations * batch_size

    labeled_sampler = RandomSampler(labeled_dataset, replacement=True, num_samples=total_labeled_samples)
    unlabeled_sampler = RandomSampler(unlabeled_dataset, replacement=True, num_samples=total_unlabeled_samples)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, sampler=labeled_sampler)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, sampler=unlabeled_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return labeled_loader, unlabeled_loader, test_loader

# MixUp数据增强
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam

def sharpen(p, T=0.5):
    temp = p ** (1 / T)
    temp /= temp.sum(dim=1, keepdim=True)
    return temp

def guess_labels(model, unlabeled_data, num_augmentations, device):
    """使用模型预测为无标签数据猜测标签。"""
    model.eval()
    with torch.no_grad():
        outputs = [model(unlabeled_data.to(device)) for _ in range(num_augmentations)]
        outputs = torch.stack(outputs)
        avg_predictions = outputs.mean(0)
        sharpened_predictions = sharpen(avg_predictions, T=0.5)
        
        # 统计伪标签分布
        pseudo_labels = sharpened_predictions.argmax(dim=1).cpu().numpy()
        unique, counts = np.unique(pseudo_labels, return_counts=True)
#         print("Pseudo-label distribution: ", dict(zip(unique, counts)))
        
        return sharpened_predictions
    
# 定义WideResNet模型
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], 10)
        self.nChannels = nChannels[3]

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

# 预训练函数
def pretrain_model(model, labeled_loader, device, epochs=1):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for labeled_data, labels in labeled_loader:
            labeled_data, labels = labeled_data.to(device), labels.to(device)

            # 前向传播
            logits = model(labeled_data)
            loss = criterion(logits, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Pretrain Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(labeled_loader):.4f}")

def train(model, labeled_loader, unlabeled_loader, test_loader, device, epochs=1):
    optimizer = Adam(model.parameters(), lr=0.00001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    iter_loss = []
    iter_accuracy = []
    iteration_count = 0
    loss_per_iteration = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        unlabeled_iter = iter(unlabeled_loader)
        
        for labeled_data, labels in tqdm(labeled_loader):
            try:
                unlabeled_data, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_data, _ = next(unlabeled_iter)

            # 确保有标签和无标签数据的批次大小相同
            min_batch_size = min(labeled_data.size(0), unlabeled_data.size(0))
            labeled_data = labeled_data[:min_batch_size].to(device)
            labels = labels[:min_batch_size].to(device)
            unlabeled_data = unlabeled_data[:min_batch_size].to(device)

            # 猜测无标签数据的标签
            pseudo_labels = guess_labels(model, unlabeled_data, num_augmentations=5, device=device)

            # 将labels进行one-hot编码
            one_hot_labels = F.one_hot(labels, num_classes=10).float()

            # 继续使用mixed_data和mixed_labels
            all_data = torch.cat([labeled_data, unlabeled_data], dim=0)
            all_labels = torch.cat([one_hot_labels, pseudo_labels], dim=0)

            mixed_data, mixed_labels, lam = mixup_data(all_data, all_labels, alpha=0.75, use_cuda=(device == 'cuda'))

            # 前向传播
            logits = model(mixed_data)
            loss = criterion(logits, mixed_labels.argmax(dim=1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iteration_count += 1
            
            # 每100次迭代记录一次损失
            if iteration_count % 100 == 0:
                loss_per_iteration.append(loss.item())

            # 每1000次迭代测试一次正确率
            if iteration_count % 1000 == 0:
                model.eval()
                total = 0
                correct = 0
                with torch.no_grad():
                    for data, labels in test_loader:
                        data, labels = data.to(device), labels.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                iteration_accuracy = (correct / total) * 100
                iter_loss.append((iteration_count, epoch_loss / iteration_count))
                iter_accuracy.append((iteration_count, iteration_accuracy))
                print(f'Iteration [{iteration_count}], Loss: {epoch_loss / iteration_count:.4f}, Accuracy: {iteration_accuracy:.2f}%')
                model.train()

        # 每轮结束后绘制图表
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss_per_iteration) + 1), loss_per_iteration, label='Loss per 1000 Iterations')
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.legend()

        plt.subplot(1, 2, 2)
        if iter_accuracy:
            plt.plot(*zip(*iter_accuracy), label='Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy over Iterations')
        plt.legend()

        plt.savefig(f'mixmatch-4000.png')
        # plt.show()
        
# 设置设备并运行训练过程
device = 'cuda' if torch.cuda.is_available() else 'cpu'
labeled_loader, unlabeled_loader, test_loader = load_cifar10(batch_size=64, num_labeled_samples=4000, num_iterations=20000)
model = WideResNet(depth=28, widen_factor=2, dropRate=0.3)

# 预训练模型
# pretrain_model(model, labeled_loader, device, epochs=10)

# 使用无标签数据进行训练
train(model, labeled_loader, unlabeled_loader, test_loader, device, epochs=1)
