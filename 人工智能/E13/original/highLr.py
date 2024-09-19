# origin version
# but I use the preprocessed data
# got overfit when the lr is set to 0.001, but nice performance
# run the code: python origin.py --dropout 0

import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

class QNLIDataset(Dataset):
    def __init__(self, file_path, max_length, embedding_model):
        self.sentences = [] # each element is vectors of words in the sentense
        self.labels = []
        self.max_length = max_length
        self.embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]: # 0 line need to be omitted
            items = line.strip().split('\t') # index
            premise = items[1].split() # question
            hypothesis = items[2].split() # sentense
            sentence = premise + hypothesis # so no devision
            sentence = sentence[:self.max_length] # cut it
            
            sentence_vectors = [] # each element is the vector of each word
            for word in sentence:
                if word in embedding_model:
                    sentence_vectors.append(embedding_model[word])
            if len(sentence_vectors) == 0:
                sentence_vectors.append(np.zeros(self.embedding_dim))
            while len(sentence_vectors) < self.max_length: # padding
                sentence_vectors.append(np.zeros_like(sentence_vectors[0]))
            self.sentences.append(np.array(sentence_vectors))
            self.labels.append(int(items[3] == 'entailment'))

            # need more process? padding?
    
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
        #  (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LSTMModel_d(nn.Module):
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

def train(model, data_loader):
    model.train()
    running_loss = 0.
    correct = 0
    total = 0
    for inputs, labels in data_loader: # __getitem__
        inputs = inputs
        labels = labels
        optimizer.zero_grad() # or will be influenced by backward()
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
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", default=False, type=int, help="use dropout or not")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    args = parser.parse_args()
    
    print(f"dropout is set? {args.dropout}")

    embedding_model = {} # is a dictionary
    with open('../dataset/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            embedding_model[word] = vector
    # get the len of the first value -> 50 if using 50d.txt
    embedding_dim = len(embedding_model[next(iter(embedding_model.keys()))])

    train_file = '../dataset/train_40.tsv'
    valid_file = '../dataset/dev_40.tsv'
    # train_file = '../dataset/train_40_data_preprocessed.tsv'
    # valid_file = '../dataset/dev_40_data_preprocessed.tsv'

    max_length = 50
    batch_size = 32

    train_dataset = QNLIDataset(train_file, max_length, embedding_model)
    valid_dataset = QNLIDataset(valid_file, max_length, embedding_model)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(valid_dataset))

    input_size = embedding_dim
    hidden_size = 64
    num_layers = 1
    output_size = 2

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    if args.dropout:
        model = LSTMModel_d(input_size, hidden_size, num_layers, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.lr)

    loss=[]
    accuracy=[]

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
    # plt.show()
    if args.dropout:
        plt.savefig('../result/dropout_acc.png')
        plt.clf()
    else:
        plt.savefig('../result/ori_acc.png')
        plt.clf()

    plt.plot([i+1 for i in range(len(loss))],loss,color="b")
    plt.title("loss")
    plt.xlabel('epoch')
    plt.ylabel('average loss') 
    # plt.show()
    if args.dropout:
        plt.savefig('../result/dropout_loss.png')
        plt.clf()
    else:
        plt.savefig('../result/origin_loss.png')
        plt.clf()