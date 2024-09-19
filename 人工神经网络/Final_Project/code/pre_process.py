import json
import re
import nltk
import jieba
from collections import Counter
import numpy as np
import random
import torch
from collections import Counter

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def clean_text(text):
    # 将中文标点转换为英文标点
    text = text.replace('，', ',').replace('。', '.').replace('；', ';').replace('：', ':').replace('？', '?').replace('！', '!')
    # 去除异常字符（非字母数字和空格）
    text = re.sub(r'[^\w\s,.!?]', '', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 转换为小写
    text = text.lower()
    return text.strip()

# 读取和清洗数据
def read_and_clean_data(filepath):
    cleaned_data = []
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            item['en'] = clean_text(item['en'])
            item['zh'] = clean_text(item['zh'])
            cleaned_data.append(item)
    return cleaned_data


# 过滤超长句子
def filter_sentences(sentences, max_length=50):
    return [sentence for sentence in sentences if len(sentence['en_tokens']) < max_length and
            len(sentence['zh_tokens']) < max_length
            ]


# 分词
def tokenize_text(data):
    nltk.download('punkt')
    for item in data:
        item['en_tokens'] = nltk.word_tokenize(item['en'])
        item['zh_tokens'] = list(jieba.cut(item['zh']))
    return data

def build_dict(sentences_token, name, max_size=10000):
    lang = Lang(name)
    word_counter = Counter()

    for sentence in sentences_token:
        word_counter.update(sentence)

    # 获取词频前 max_size 的单词
    most_common_words = word_counter.most_common(max_size)

    for word, count in most_common_words:
        for _ in range(count):
            lang.addWord(word)

    return lang


def pre_process(data_filepath):
    # 读取和清洗数据
    cleaned_data = read_and_clean_data(data_filepath)
    # 分词
    tokenized_data = tokenize_text(cleaned_data)
    # 过滤句子长度超过50的数据
    filtered_data = filter_sentences(tokenized_data, max_length=50)

    tokenized_en = [item['en_tokens'] for item in filtered_data]
    tokenized_zh = [item['zh_tokens'] for item in filtered_data]
    
    english_lang = build_dict(tokenized_en, "en",10000)
    chinese_lang = build_dict(tokenized_zh, "zh",10000)
        
    return filtered_data,english_lang,chinese_lang

