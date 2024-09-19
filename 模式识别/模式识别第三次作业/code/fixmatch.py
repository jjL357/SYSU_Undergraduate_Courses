import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
import random
from tqdm import tqdm
from WideResNet import WideResNet 
import matplotlib.pyplot as plt

# 设置随机种子以确保结果的可重复性
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 定义数据增强
weak_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

strong_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 定义一个自定义数据集类以应用转换
class CustomDataset(Dataset):
    def __init__(self, dataset, weak_transform=None, strong_transform=None):
        self.dataset = dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.weak_transform and self.strong_transform:
            img_weak = self.weak_transform(img)
            img_strong = self.strong_transform(img)
            return img_weak, img_strong, label
        elif self.weak_transform:
            img = self.weak_transform(img)
            return img, label
        return img, label

# 加载CIFAR-10数据集
def get_cifar10_datasets():
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    return train_dataset, test_dataset

# 获取标记和未标记的数据集，确保有标签数据从10个类别中随机挑选
def get_semi_supervised_datasets(train_dataset, num_labeled):
    labeled_indices = []
    unlabeled_indices = []
    
    # 创建一个字典用于存储每个类别的索引列表
    class_indices = {i: [] for i in range(10)}
    
    # 将训练集中每个类别的索引分别存储到字典中
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    # 随机选择 num_labeled 个样本作为有标签数据
    labeled_indices = []
    for label in range(10):
        indices = class_indices[label]
        random.shuffle(indices)
        labeled_indices.extend(indices[:num_labeled//10])
        unlabeled_indices.extend(indices[num_labeled//10:])
    
    labeled_dataset = CustomDataset(Subset(train_dataset, labeled_indices), weak_transform=weak_transform)
    unlabeled_dataset = CustomDataset(Subset(train_dataset, unlabeled_indices), weak_transform=weak_transform, strong_transform=strong_transform)
    
    return labeled_dataset, unlabeled_dataset

# WideResNet-28-2定义
class WideResNet28_2(nn.Module):
    def __init__(self):
        super(WideResNet28_2, self).__init__()
        self.model = WideResNet(depth=28, widen_factor=2, dropRate=0.3, num_classes=10)

    def forward(self, x):
        return self.model(x)

def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, criterion, device, lambda_u=1.0, tau=0.95, test_loader=None):
    model.train()
    
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    labeled_epoch = 0
    unlabeled_epoch = 0
    
    for batch_idx in tqdm(range(1024), desc='Training', leave=False):
        try:
            inputs_x, targets_x = labeled_iter.__next__()
        except StopIteration:
            labeled_epoch += 1
            labeled_loader = DataLoader(labeled_loader.dataset, batch_size=labeled_loader.batch_size, shuffle=True, num_workers=labeled_loader.num_workers)
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = labeled_iter.__next__()

        try:
            inputs_u_w, inputs_u_s, _ = unlabeled_iter.__next__()
        except StopIteration:
            unlabeled_epoch += 1
            unlabeled_loader = DataLoader(unlabeled_loader.dataset, batch_size=unlabeled_loader.batch_size, shuffle=True, num_workers=unlabeled_loader.num_workers)
            unlabeled_iter = iter(unlabeled_loader)
            inputs_u_w, inputs_u_s, _ = unlabeled_iter.__next__()
        
        inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
        inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)
        
        # 模型预测
        logits_x = model(inputs_x)
        logits_u_w = model(inputs_u_w)
        
        # 生成伪标签
        pseudo_labels = torch.softmax(logits_u_w, dim=-1)
        max_probs, targets_u = torch.max(pseudo_labels, dim=-1)
        
        # 过滤掉低置信度的伪标签
        mask = max_probs.ge(tau).float()
        
        # 计算有监督损失
        loss_x = criterion(logits_x, targets_x)
        
        # 计算无监督损失
        logits_u_s = model(inputs_u_s)
        loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        # 总损失
        loss = loss_x + lambda_u * loss_u
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 输出测试集的准确率
        if test_loader is not None and batch_idx % 100 == 0:
            test_loss, test_acc = test(model, test_loader, device)
            print(f'Batch {batch_idx}, Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}')

# 定义测试函数
def test(model, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# 保存日志
def save_log(epoch, train_loss, train_acc, test_loss, test_acc, log_file='log.txt'):
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch}:\n')
        f.write(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%\n')
        f.write(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n')
        f.write('\n')

# 保存可视化图片
def save_plot(epochs, train_losses, train_accuracies, test_losses, test_accuracies, plot_file='training_plot.png'):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

# 主函数
def main():
    # 参数设置
    num_labeled = 4000
    batch_size = 40
    num_epochs = 20
    learning_rate = 0.03
    weight_decay = 0.0005  # 设置 weight decay
    
    # 获取数据集
    train_dataset, test_dataset = get_cifar10_datasets()
    labeled_dataset, unlabeled_dataset = get_semi_supervised_datasets(train_dataset, num_labeled)
    
    # 数据加载器
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size * 7, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 初始化模型、损失函数和优化器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WideResNet28_2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用 SGD 优化器
    
    # 记录训练和测试的损失和准确率
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    epochs = list(range(1, num_epochs + 1))
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # 训练模型
        train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, criterion, device, test_loader=test_loader)
        
        # 计算训练集上的损失和准确率
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs_x, targets_x in labeled_loader:
            inputs_x, targets_x = inputs_x.to(device), targets_x.to(device)
            
            # 前向传播
            outputs = model(inputs_x)
            loss = criterion(outputs, targets_x)
            
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += targets_x.size(0)
            correct_train += (predicted == targets_x).sum().item()
        
        avg_train_loss = total_train_loss / len(labeled_loader)
        train_accuracy = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        
        # 计算测试集上的损失和准确率
        test_loss, test_accuracy = test(model, test_loader, device)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # 保存日志
        save_log(epoch, avg_train_loss, train_accuracy, test_loss, test_accuracy)
        
        # 打印日志
        print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    # 保存可视化图片
    save_plot(epochs, train_losses, train_accuracies, test_losses, test_accuracies)

if __name__ == '__main__':
    main()
