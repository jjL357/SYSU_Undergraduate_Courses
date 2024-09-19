import torch
import torch.nn as nn
from config import my_config


class myLSTM(nn.Module):
    def __init__(self, vocab_size, config: my_config):
        super(myLSTM, self).__init__()  # 初始化
        self.vocab_size = vocab_size
        self.config = config
        self.embeddings = nn.Embedding(vocab_size, self.config.embedding_size)  # 配置嵌入层，计算出词向量
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size,  # 输入大小为转化后的词向量
            hidden_size=self.config.hidden_size,  # 隐藏层大小
            num_layers=self.config.num_layers,  # 堆叠层数，有几层隐藏层就有几层
            dropout=self.config.dropout,  # 遗忘门参数
            bidirectional=True  # 双向LSTM
        )

        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(
            self.config.num_layers * self.config.hidden_size * 2,  # 因为双向所有要*2
            self.config.output_size
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedded = self.embeddings(x)

        lstm_out, (h_n, c_n) = self.lstm(embedded)

        feature = self.dropout(h_n)

        # 这里将所有隐藏层进行拼接来得出输出结果，没有使用模型的输出
        feature_map = torch.cat([feature[i, :, :] for i in range(feature.shape[0])], dim=-1)

        out = self.fc(feature_map)

        return self.softmax(out)
