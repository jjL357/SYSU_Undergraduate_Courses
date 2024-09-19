import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
import time

# 定义文件路径
output_file = 'training_log.txt'
checkpoint_path = 'fer_cnn_checkpoint.pth'

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小为 48x48
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化
])

# 定义模型
class FER_CNN(nn.Module):
    def __init__(self):
        super(FER_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(num_features=128),
           nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(512*3*3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
           nn.ReLU(inplace=True),
            nn.Linear(256, 5),

        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)
        return y

def load_data(train_dir, test_dir, batch_size, val_split=0.2):
    # 加载训练数据集
    train_dataset = ImageFolder(root=train_dir, transform=transform)

    # 定义验证集比例
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(np.floor(val_split * dataset_size))
    train_indices, val_indices = indices[split:], indices[:split]

    # 定义训练集和验证集的采样器
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 加载训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)

    # 加载测试数据集
    test_dataset = ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, output_file='training_log.txt', checkpoint_path='checkpoint.pth'):
    start_time = time.time()
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    bad_epochs = 0

    with open(output_file, 'w') as f:
        for epoch in range(num_epochs):
            f.write(f'Epoch [{epoch+1}/{num_epochs}]\n')
            print(f'Epoch [{epoch+1}/{num_epochs}]\n')

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            f.write(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n')
            print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%\n')

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            f.write(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n')

            if val_loss > best_val_loss or val_accuracy < best_val_accuracy:
                f.write("Validation loss increased or accuracy decreased.\n")
                print("Validation loss increased or accuracy decreased.\n")
                bad_epochs += 1
                if bad_epochs >= 3:
                    f.write("Stopping training due to consecutive bad epochs.\n")
                    print("Stopping training due to consecutive bad epochs.\n")

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy
                    }, checkpoint_path)

                    break
            else:
                bad_epochs = 0

            best_val_loss = val_loss
            best_val_accuracy = val_accuracy

        end_time = time.time()
        elapsed_time = end_time - start_time
        f.write(f'Training time: {elapsed_time:.2f} seconds\n')
        print(f'Training time: {elapsed_time:.2f} seconds\n')

def test_model(model, test_loader, criterion, output_file='training_log.txt'):
    with open(output_file, 'a') as f:
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        test_class_correct = list(0. for i in range(5))
        test_class_total = list(0. for i in range(5))
        with torch.no_grad():
            for images, labels in tqdm(test_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    test_class_correct[label] += c[i].item()
                    test_class_total[label] += 1
        
        test_loss /= len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        f.write(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%\n')

        for i in range(5):
            recall = 100 * test_class_correct[i] / test_class_total[i]
            precision = 100 * test_class_correct[i] / test_total
            f1 = 2 * (precision * recall) / (precision + recall)
            f.write(f'Class {i} - Recall: {recall:.2f}%, Precision: {precision:.2f}%, F1: {f1:.2f}\n')
            print(f'Class {i} - Recall: {recall:.2f}%, Precision: {precision:.2f}%, F1: {f1:.2f}\n')


# 定义路径
train_dir = 'D://d_code//git//人工神经网络//MidTerm//mid_hw_emotion_recognition//train'
test_dir = 'D://d_code//git//人工神经网络//MidTerm//mid_hw_emotion_recognition//test'

# 加载数据
train_loader, val_loader, test_loader = load_data(train_dir, test_dir, batch_size=64)

# 定义模型、损失函数、优化器
model = FER_CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
train_model(model, train_loader, val_loader, criterion, optimizer, output_file=output_file, checkpoint_path=checkpoint_path)

# 测试模型
test_model(model, test_loader, criterion, output_file=output_file)

