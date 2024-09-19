# I don't run this on my pc
# I modify it into ipynb and run in colab

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import time

class QNLIDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.sentences = []
        self.labels = []
        self.max_length = max_length

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Skip header

        for line in lines:
            items = line.strip().split('\t')
            premise = items[1]
            hypothesis = items[2]
            # this sign
            sentence = "[CLS] " + premise + " [SEP] " + hypothesis + " [SEP]"
            encoded_dict = tokenizer.encode_plus(
                sentence,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            self.sentences.append((encoded_dict['input_ids'], encoded_dict['attention_mask']))
            self.labels.append(int(items[3] == 'entailment'))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

class BERTModel(nn.Module):
    def __init__(self, output_size, pretrained_model_name='bert-base-uncased'):
        super(BERTModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropped_out = self.dropout(pooled_output)
        return self.fc(dropped_out)

def train(model, data_loader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in data_loader:
        (input_ids, attention_mask), labels = batch
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.squeeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * input_ids.size(0)
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
        for batch in data_loader:
            (input_ids, attention_mask), labels = batch
            input_ids = input_ids.squeeze(1).to(device)
            attention_mask = attention_mask.squeeze(1).to(device)
            labels = labels.to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

if __name__ == "__main__":
    pretrained_model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    train_file = '../dataset/train_40.tsv'
    valid_file = '../dataset/dev_40.tsv'

    max_length = 50
    batch_size = 16

    train_dataset = QNLIDataset(train_file, tokenizer, max_length)
    valid_dataset = QNLIDataset(valid_file, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    print('Number of training samples:', len(train_dataset))
    print('Number of validation samples:', len(valid_dataset))

    output_size = 2
    num_epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERTModel(output_size, pretrained_model_name).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    loss = []
    accuracy = []

    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, device)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
        end_time = time.time()
        loss.append(valid_loss)
        accuracy.append(valid_acc)
        elapsed_time = end_time - start_time
        print('Elapsed Time:', elapsed_time, 's')
