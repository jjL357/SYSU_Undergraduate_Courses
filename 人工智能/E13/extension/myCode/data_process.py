import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# inputFlie = '../dataset/train_40.tsv'
# outputFile = '../dataset/train_40_data_preprocessed.tsv'
inputFlie = '../dataset/dev_40.tsv'
outputFile = '../dataset/dev_40_data_preprocessed.tsv'

# 下载必要的NLTK资源
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 初始化词形还原器和停用词
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# 加载QNLI数据集
data = pd.read_csv(inputFlie, sep='\t', on_bad_lines='skip') 

# 定义预处理函数
def preprocess_text(text):
    # 标记化
    words = word_tokenize(text)
    
    # 转小写、去除标点符号、去除停用词、词形还原
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
    
    return ' '.join(words)

# 对数据集中的句子进行预处理
data['question'] = data['question'].apply(preprocess_text)
data['sentence'] = data['sentence'].apply(preprocess_text)

print(data.head())
data.to_csv(outputFile, sep='\t', index=False)
