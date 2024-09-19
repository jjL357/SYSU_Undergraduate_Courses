class my_config():
    max_length = 20  # 每句话截断长度
    batch_size = 64  # 一个batch的大小
    embedding_size = 50  # 词向量大小
    hidden_size = 128  # 隐藏层大小
    num_layers = 2  # 网络层数
    dropout = 0.5  # 遗忘程度
    output_size = 2  # 输出大小
    lr = 0.001  # 学习率
    epoch = 5  # 训练次数