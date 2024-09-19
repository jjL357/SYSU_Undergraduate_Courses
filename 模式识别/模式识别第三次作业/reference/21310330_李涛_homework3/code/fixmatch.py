import random
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, RandomSampler
from torchvision import  transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

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

# 数据增强
transform_weak = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])

transform_strong = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def load_cifar10(batch_size, num_labeled_samples, num_iterations):
    transform_weak = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar10_full = CIFAR10(root='./data', train=True, transform=transform_weak, download=False)
    cifar10_test = CIFAR10(root='./data', train=False, transform=transform_test, download=False)

    indices = list(range(len(cifar10_full)))
    random.shuffle(indices)

    labeled_indices = indices[:num_labeled_samples]
    unlabeled_indices = indices[num_labeled_samples:]

    unlabeled_indices_weak = []
    unlabeled_indices_strong = []

    for idx in unlabeled_indices:
        # 随机选择每个样本是弱增强还是强增强
        if random.random() < 0.5:
            unlabeled_indices_weak.append(idx)
        else:
            unlabeled_indices_strong.append(idx)

    labeled_dataset = Subset(cifar10_full, labeled_indices)
    unlabeled_dataset_weak = Subset(cifar10_full, unlabeled_indices_weak)
    unlabeled_dataset_strong = Subset(cifar10_full, unlabeled_indices_strong)
    
    unlabeled_dataset_weak.dataset.transform = transform_weak
    unlabeled_dataset_strong.dataset.transform = transform_strong

    total_labeled_samples = num_iterations * batch_size
    total_unlabeled_samples_weak = num_iterations * batch_size
    total_unlabeled_samples_strong = num_iterations * batch_size

    labeled_sampler = RandomSampler(labeled_dataset, replacement=True, num_samples=total_labeled_samples)
    unlabeled_weak_sampler = RandomSampler(unlabeled_dataset_weak, replacement=True, num_samples=total_unlabeled_samples_weak)
    unlabeled_strong_sampler = RandomSampler(unlabeled_dataset_strong, replacement=True, num_samples=total_unlabeled_samples_strong)

    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, sampler=labeled_sampler)
    unlabeled_weak_loader = DataLoader(unlabeled_dataset_weak, batch_size=batch_size, sampler=unlabeled_weak_sampler)
    unlabeled_strong_loader = DataLoader(unlabeled_dataset_strong, batch_size=batch_size, sampler=unlabeled_strong_sampler)

    test_loader = DataLoader(cifar10_test, batch_size=64, shuffle=False)

    return labeled_loader, unlabeled_weak_loader, unlabeled_strong_loader, test_loader


def train(model, labeled_loader, unlabeled_weak_loader, unlabeled_strong_loader, test_loader, device, epochs=1):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    iteration_count = 0
    loss_per_iteration = []
    iter_accuracy = []

    best_accuracy = 0
    best_epoch = 0
    lambda_u = 1
    threshold = 0.95

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        labeled_iter = iter(labeled_loader)
        unlabeled_weak_iter = iter(unlabeled_weak_loader)
        unlabeled_strong_iter = iter(unlabeled_strong_loader)

        for _ in tqdm(range(len(labeled_loader))):
            try:
                labeled_data = next(labeled_iter)
                unlabeled_weak_data = next(unlabeled_weak_iter)
                unlabeled_strong_data = next(unlabeled_strong_iter)
            except StopIteration:
                break

            x_l, y_l = labeled_data
            x_ul_weak, _ = unlabeled_weak_data
            x_ul_strong, _ = unlabeled_strong_data

            x_l, y_l = x_l.to(device), y_l.to(device)
            x_ul_weak, x_ul_strong = x_ul_weak.to(device), x_ul_strong.to(device)

            # 监督损失
            outputs_l = model(x_l)
            loss_l = criterion(outputs_l, y_l)

            # 生成伪标签
            with torch.no_grad():
                outputs_ul_weak = model(x_ul_weak)
                pseudo_labels = torch.softmax(outputs_ul_weak, dim=1)
                max_probs, targets_ul = torch.max(pseudo_labels, dim=1)
                mask = max_probs.ge(threshold).float()

            # 一致性损失
            outputs_ul_strong = model(x_ul_strong)
            loss_u = (criterion(outputs_ul_strong, targets_ul) * mask).mean()

            train_loss = loss_l + lambda_u * loss_u

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item()
            iteration_count += 1
            
            # 每100次迭代记录一次损失
            if iteration_count % 100 == 0:
                loss_per_iteration.append(train_loss.item())

            # 每1000次迭代测试一次
            if iteration_count % 1000 == 0:
                model.eval()
                correct = 0
                total = 0
                test_loss = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)
                        loss = criterion(outputs, y)
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()

                test_accuracy = 100 * correct / total
                iter_accuracy.append((iteration_count, test_accuracy))

                print(f'Iteration [{iteration_count}], Loss: {total_loss / iteration_count:.4f}, Accuracy: {test_accuracy:.2f}%')
                model.train()

        # 每轮结束后绘制图表
        plt.figure(figsize=(12, 5))
        # 绘制训练损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(loss_per_iteration) + 1), loss_per_iteration, label='Loss per 1000 Iterations')
        plt.xlabel('Iterations (x100)')
        plt.ylabel('Loss')
        plt.title('Loss over Iterations')
        plt.legend()
        # 绘制测试准确度曲线
        plt.subplot(1, 2, 2)
        if iter_accuracy:
            plt.plot(*zip(*iter_accuracy), label='Test Accuracy')
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy over Iterations')
        plt.legend()

        plt.show()
        plt.savefig(f'fixmatch-40.png')



# 设置设备并运行训练过程
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labeled_loader, unlabeled_weak_loader, unlabeled_strong_loader, test_loader = load_cifar10(batch_size=64, num_labeled_samples=40,num_iterations=20000)

# 初始化模型
model = WideResNet(depth=28, widen_factor=2, dropRate=0.3)

train(model, labeled_loader, unlabeled_weak_loader, unlabeled_strong_loader, test_loader,device, epochs=1)
