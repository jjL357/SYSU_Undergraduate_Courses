# a litte data process 
# add dropout + add a layer

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from torch.utils.data import Dataset
import re

class QNLIDataset(Dataset):
    def __init__(self, file_path, max_length, embedding_model):
        self.sentences = []  # Each element is a list of word vectors
        self.labels = []
        self.max_length = max_length
        self.embedding_dim = len(embedding_model[next(iter(embedding_model))])
        
        special_char_pattern = re.compile(r'[^a-zA-Z0-9\s]') # special-> just replace
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip the header line

        for line in lines:
            items = line.strip().split('\t')
            premise = items[1].split()
            hypothesis = items[2].split()
            sentence = premise + hypothesis

            # Handle special characters in the sentence
            sentence = [special_char_pattern.sub('', word) for word in sentence]
            sentence = sentence[:self.max_length]  # Truncate to max_length
            
            sentence_vectors = [
                embedding_model[word] if word in embedding_model else np.zeros(self.embedding_dim)
                for word in sentence
            ]
            
            # Padding if sentence length is less than max_length
            padding_length = max(0, self.max_length - len(sentence_vectors))
            sentence_vectors.extend([np.zeros(self.embedding_dim)] * padding_length)

            self.sentences.append(np.array(sentence_vectors))
            self.labels.append(int(items[3] == 'entailment'))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sentences[idx]).float(), torch.tensor(self.labels[idx])

# 权重初始化函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.apply(init_weights)  # 初始化权重
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

def train(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
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

def evaluate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    embedding_model = {}
    with open('../dataset/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_model[word] = vector
    embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

    train_file = '../dataset/train_40.tsv'
    valid_file = '../dataset/dev_40.tsv'

    max_length = 50
    batch_size = 32

    train_dataset = QNLIDataset(train_file, max_length, embedding_model)
    valid_dataset = QNLIDataset(valid_file, max_length, embedding_model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(valid_dataset))

    input_size = embedding_dim
    hidden_size = 128  # 增加隐藏单元数
    num_layers = 2  # 增加LSTM层数
    output_size = 2
    num_epochs = 10  # 增加训练周期

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 0.0001

    loss = []
    accuracy = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Valid Loss: {:.4f}, Valid Acc: {:.4f}'.format(
            epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))
        end_time = time.time()
        loss.append(valid_loss)
        accuracy.append(valid_acc)
        elapsed_time = end_time - start_time
        print('Elapsed Time:', elapsed_time, 's')

    plt.plot([i+1 for i in range(len(accuracy))],accuracy,color="g")
    plt.title("Accuracy")
    plt.xlabel('epoch')
    plt.ylabel('accuracy') 
    # plt.show()
    plt.savefig('../result/opt_acc.png')
    plt.clf()

    plt.plot([i+1 for i in range(len(loss))],loss,color="b")
    plt.title("loss")
    plt.xlabel('epoch')
    plt.ylabel('average loss') 
    plt.savefig('../result/opt_loss.png')
    # plt.show()
    plt.clf()