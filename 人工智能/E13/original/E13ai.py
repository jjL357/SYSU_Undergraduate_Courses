# coding=gb2312
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import time
from gensim.models import KeyedVectors
import numpy as np

# 加载GloVe词向量
def load_custom_word_vectors(file_path):
    word_vectors = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.strip().split()
            word = values[0]
            vector = np.array([float(val) for val in values[1:]])
            word_vectors[word] = vector

    return word_vectors

# 使用GloVe词向量生成嵌入矩阵
def generate_embedding_matrix(word_vectors):
    embedding_dim = len(next(iter(word_vectors.values())))
    embedding_matrix = np.zeros((len(word_vectors), embedding_dim))
    word_index = {}

    index = 0
    for i, word in enumerate(word_vectors.keys()):
        word_index[word] = index
        if word in word_vectors:
            embedding_matrix[index] = word_vectors[word]
            index += 1

    return embedding_matrix, word_index

# 定义模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        output = self.fc(h_n[-1])
        return output

# 加载数据集
def load_dataset(file_path):
    texts = []
    labels = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if parts[0] == 'index':
                continue
            text = parts[1] + " " + parts[2]
            label = int(parts[3] == 'entailment')
            texts.append(text)
            labels.append(label)

    return texts, labels

# 将文本转换为词向量索引序列
def text_to_sequence(texts, word_index, max_length):
    sequences = []

    for text in texts:
        words = text.split()
        sequence = []
        for word in words:
            if word not in word_index:
                continue
            index = word_index[word]
            sequence.append(index)
        sequence += [0] * (max_length - len(sequence))
        sequences.append(sequence)

    return sequences

# 创建自定义Dataset类
class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        label = self.labels[index]
        return sequence, label

# 参数设置
input_size = 50  # 词向量维度
hidden_size = 64
num_classes = 2
batch_size = 32
num_epochs = 10

# 加载GloVe词向量
word_vectors = load_custom_word_vectors('E13/glove.6B.50d.txt')

# 使用GloVe词向量生成嵌入矩阵和词索引
embedding_matrix, word_index = generate_embedding_matrix(word_vectors)

# 创建模型
model = LSTMClassifier(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 加载数据集
train_texts, train_labels = load_dataset('E13/QNLI/train_40.tsv')
test_texts, test_labels = load_dataset('E13/QNLI/dev_40.tsv')

max_length = 50
# 将文本转换为词向量索引序列
train_sequences = text_to_sequence(train_texts, word_index, max_length)
test_sequences = text_to_sequence(test_texts, word_index, max_length)

# 创建数据加载器
train_sequences = torch.tensor(train_sequences, dtype=torch.long)
test_sequences = torch.tensor(test_sequences, dtype=torch.long)

train_dataset = CustomDataset(train_sequences, train_labels)
test_dataset = CustomDataset(test_sequences, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
total_step = len(train_loader)
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%')

end_time = time.time()
elapsed_time = end_time - start_time
print(f'Training Time: {elapsed_time:.2f} seconds')