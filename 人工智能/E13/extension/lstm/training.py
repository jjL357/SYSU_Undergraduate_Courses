from dataset import mydata, Dataset
from model import myLSTM
from config import my_config
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn):  # 训练模型
    '''
    :param model:模型
    :param train_iterator:训练数据的迭代器
    :param dev_iterator: 验证数据的迭代器
    :param optimzer: 优化器
    :param loss_fn: 损失函数
    '''
    losses = []
    for i, batch in enumerate(train_iterator):
        if torch.cuda.is_available():
            input = batch.text.cuda()
            label = batch.label.type(torch.cuda.LongTensor)
        else:
            input = batch.text
            label = batch.label

        pred = model(input)  # 预测
        loss = loss_fn(pred, label)  # 计算损失值
        loss.backward()  # 误差反向传播
        losses.append(loss.data.numpy())  # 记录误差
        optimzer.step()  # 优化一次

        # if i % 30 == 0:  # 训练30个batch后查看损失值和准确率
        #     avg_train_loss = np.mean(losses)
        #     print(f'iter:{i + 1},avg_train_loss:{avg_train_loss:.4f}')
        #     losses = []
        #     val_acc = evaluate_model(model, dev_iterator)
        #     print('val_acc:{:.4f}'.format(val_acc))
        #     model.train()


def evaluate_model(model, dev_iterator):  # 评价模型
    '''
    :param model:模型
    :param dev_iterator:待评价的数据
    :return:评价（准确率）
    '''
    all_pred = []
    all_y = []
    for i, batch in enumerate(dev_iterator):
        if torch.cuda.is_available():
            input = batch.text.cuda()
            label = batch.label.type(torch.cuda.LongTensor)
        else:
            input = batch.text
            label = batch.label

        y_pred = model(input)  # 预测
        predicted = torch.max(y_pred.cpu().data, 1)[1]  # 选择概率最大作为当前数据预测结果
        all_pred.extend(predicted.numpy())
        all_y.extend(label.numpy())
    score = accuracy_score(all_y, np.array(all_pred).flatten())  # 计算准确率
    return score


if __name__ == '__main__':
    config = my_config()  # 配置对象实例化
    data_class = mydata()  # 数据类实例化
    config.output_size = data_class.n_class
    dataset = Dataset(data_class, config)  # 数据预处理实例化

    dataset.load_data()  # 进行数据预处理

    train_iterator = dataset.train_iterator  # 得到处理好的数据迭代器
    dev_iterator = dataset.dev_iterator
    # test_iterator = dataset.test_iterator

    vocab_size = len(dataset.vocab)  # 字典大小

    # 初始化模型
    model = myLSTM(vocab_size, config)
    model.embeddings.weight.data.copy_(dataset.pretrained_embedding)  # 使用训练好的词向量初始化embedding层

    optimzer = torch.optim.Adam(model.parameters(), lr=config.lr)  # 优化器
    loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失函数

    y = []
    for i in range(config.epoch):
        print(f'epoch:{i + 1}')
        run_epoch(model, train_iterator, dev_iterator, optimzer, loss_fn)

        # 训练一次后评估一下模型
        train_acc = evaluate_model(model, train_iterator)
        dev_acc = evaluate_model(model, dev_iterator)
        #test_acc = evaluate_model(model, test_iterator)

        print('#' * 20)
        print('train_acc:{:.4f}'.format(train_acc))
        print('dev_acc:{:.4f}'.format(dev_acc))
        # print('test_acc:{:.4f}'.format(test_acc))

        y.append(dev_acc)
    # 训练完画图
    x = [i for i in range(len(y))]
    fig = plt.figure()
    plt.plot(x, y)
    plt.show()
    plt.savefig('../result/lstm_acc.png')