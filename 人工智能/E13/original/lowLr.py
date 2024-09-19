# coding=gb2312
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import gensim
import matplotlib.pyplot as plt
#定义一个自定义数据集类，用于加载并预处理数据集：
class QNLIDataset(Dataset):
    def __init__(self, file_path, max_length, embedding_model):
        self.sentences = []
        self.labels = []
        self.max_length = max_length
        self.embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
                # 读取每一行数据，格式为：premise sentence  hypothesis sentence  label
                items = line.strip().split('\t')
                premise = items[1].split()
                hypothesis = items[2].split()
                # 将premise和hypothesis合并成一个句子
                sentence = premise + hypothesis
                # 限制句子最大长度为max_length
                sentence = sentence[:self.max_length]
                # 将句子中的每个词转换为词向量
               
                sentence_vectors = []
                for word in sentence:
                    if word in embedding_model:
                        sentence_vectors.append(embedding_model[word])
                    if len(sentence_vectors) == 0:
                        sentence_vectors.append(np.zeros(self.embedding_dim))
                # 将句子的长度补齐到max_length，并将句子向量作为样本
                while len(sentence_vectors) < self.max_length:
                    sentence_vectors.append(np.zeros_like(sentence_vectors[0]))
                self.sentences.append(np.array(sentence_vectors))
                # 将标签转换为0或1
                self.labels.append(int(items[3] == 'entailment'))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sentences[idx]).float(), torch.tensor(self.labels[idx])




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
'''
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out
'''
# 定义训练函数和验证函数
def train(model, data_loader):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs = inputs
        labels = labels
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, data_loader):
    model.eval()
    running_loss = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs
            labels = labels
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



# 加载GloVe词向量模型

embedding_model = {}
with open('E13\glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embedding_model[word] = vector
embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])



# 定义训练集和验证集文件路径
train_file = 'E13/QNLI/train_40.tsv'
valid_file = 'E13/QNLI/dev_40.tsv'

# 定义句子最大长度和批次大小
max_length = 50
batch_size = 32

# 加载训练集和验证集
train_dataset = QNLIDataset(train_file, max_length, embedding_model)
valid_dataset = QNLIDataset(valid_file, max_length, embedding_model)

# 定义训练集和验证集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# 输出数据集中的样本数
print('Number of training samples:', len(train_dataset))
print('Number of validation samples:', len(valid_dataset))



# 定义模型的超参数
input_size = embedding_dim
hidden_size = 64
num_layers = 1
output_size = 2


# 实例化模型、损失函数和优化器
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.0001)




loss=[]
accuracy=[]

# 开始训练和验证
num_epochs = 20
for epoch in range(num_epochs):
    start_time = time.time()
    train_loss, train_acc = train(model, train_loader)
    valid_loss, valid_acc = evaluate(model, valid_loader)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'.format(epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))
    end_time = time.time()
    loss.append(valid_loss)
    accuracy.append(valid_acc)
    elapsed_time = end_time - start_time
    print('Elapsed Time:', elapsed_time,'s')

plt.plot([i+1 for i in range(len(accuracy))],accuracy,color="g")
plt.title("Accuracy")
plt.xlabel('epoch')
plt.ylabel('accuracy') 
plt.show()

plt.plot([i+1 for i in range(len(loss))],loss,color="b")
plt.title("loss")
plt.xlabel('epoch')
plt.ylabel('average loss') 
plt.show()


