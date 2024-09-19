import pandas as pd
import os
import torchtext
from tqdm import tqdm


class mydata(object):
    def __init__(self):
        self.data_dir = '../dataset'
        self.n_class = 2

    def _generator(self, filename):  # 加载每行数据及其标签
        path = os.path.join(self.data_dir, filename)
        # need to skip some invalid line
        df = pd.read_csv(path, sep='\t', header=0, on_bad_lines='skip')
        for index, line in df.iterrows():
            # modified
            idx = line[0]
            s1 = line[1]
            s2 = line[2]
            sentence = s1 + s2
            label = int(line[3] == 'entailment')
            yield sentence, label

    def load_train_data(self):  # 加载数据
        return self._generator('train_40.tsv')

    def load_dev_data(self):
        return self._generator('dev_40.tsv')

    def load_test_data(self):
        return self._generator('dev_40.tsv') # don't have


class Dataset(object):
    def __init__(self, dataset: mydata, config):
        self.dataset = dataset
        self.config = config  # 配置文件

    def load_data(self):
        tokenizer = lambda sentence: [x for x in sentence.split() if x != ' ']  # 以空格切词
        # 定义field
        TEXT = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_length)
        LABEL = torchtext.data.Field(sequential=False, use_vocab=False)

        # text, label能取出example对应的数据
        # 相当于定义了一种数据类型吧
        datafield = [("text", TEXT), ("label", LABEL)]

        # 加载数据
        train_gen = self.dataset.load_train_data()
        dev_gen = self.dataset.load_dev_data() # I wonder how this is used 
        test_gen = self.dataset.load_test_data() # well, don't have

        # 转换数据为example对象（数据+标签）
        train_example = [torchtext.data.Example.fromlist(it, datafield) for it in tqdm(train_gen)]
        dev_example = [torchtext.data.Example.fromlist(it, datafield) for it in tqdm(dev_gen)]
        test_example = [torchtext.data.Example.fromlist(it, datafield) for it in tqdm(test_gen)]

        # 转换成dataset
        train_data = torchtext.data.Dataset(train_example, datafield)  # example, field传入
        dev_data = torchtext.data.Dataset(dev_example, datafield)
        test_data = torchtext.data.Dataset(test_example, datafield)

        # 训练集创建字典，默认添加两个特殊字符<unk>和<pad>
        TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.50d')  # max_size出现频率最高的 k 个单词，加载100d的词向量
        self.vocab = TEXT.vocab        # 获取字典
        self.pretrained_embedding = TEXT.vocab.vectors  # 保存词向量

        # 放入迭代器并打包成batch及按元素个数排序，到时候直接调用即可
        self.train_iterator = torchtext.data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            shuffle=False
        )

        self.dev_iterator = torchtext.data.BucketIterator(
            (dev_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False
        )

        # self.dev_iterator, self.test_iterator = torchtext.data.BucketIterator.splits(
        #     (dev_data, test_data),
        #     batch_size=self.config.batch_size,
        #     sort_key=lambda x: len(x.text),
        #     repeat=False
        # )

        print(f"load {len(train_data)} training examples")
        print(f"load {len(dev_data)} dev examples")
        # print(f"load {len(test_data)} test examples")