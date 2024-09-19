# Mid-term Homework:Facial Expression Recognition (FER)
## 1. 实验内容
### 1.1 任务定义域数据集
▶任务定义：对于给定的人脸图片，输出其表情标签（图像分
类任务）；
▶数据集下载地址：
https://pan.baidu.com/s/1xbCuLRwLY5GEIRHPUAhSgg?pwd=d69e
▶一共包含五种表情：Angry，Happy，Neutral，Sad，Surprise，已经进行了训练集和测试集的划分，数据规模如下（数据集中存在一定的不平衡）；


| 情绪标签 | 训练集样本数 | 测试集样本数 |
|----------|--------------|--------------|
| Angry    | 500          | 100          |
| Happy    | 1500         | 100          |
| Neutral  | 1000         | 100          |
| Sad      | 1500         | 100          |
| Surprise | 500          | 100          |

### 1.2 作业要求
▶ python 环境
- 深度学习框架使用 pytorch；
- 请提交 requirement.txt，包含必要的第三方包即可；
▶ 模型的构建和训练
- 自行设计卷积神经网络对人脸特征进行抽取，并通过全连接
层进行分类；
- 不允许加载现成的预训练模型或图像分类包；
- 可以参考经典 CNN 结构（AlexNet、VGG、Resnet 等）；
- 可以自行探索网络结构对性能的影响；
▶ 模型测试
- 使用已经划分的测试集对训练好的模型进行测试，计算准确
率，和每个类别的召回率、精准率，Macro-F1 等；
- 请勿使用测试集进行训练（作弊）；

### 1.3 提交内容
▶ 代码 +requirement.txt
▶ 训练好的 checkpoint
▶ README：使用指南（如何进行训练、评测）
▶ 文档：对你所实现的内容进行详细阐述；包括但不限于：
- 数据预处理的操作
- 超参数的设置
- 模型的架构设计
- 验证方法（划分验证集进行验证）
- 结果分析
- 自己的探索和心得体会；


### 1.4 帮助
▶ 不熟悉 pytorch 的同学可以参阅官方教程
https://pytorch.org/tutorials/
▶ 环境安装
https://pytorch.org/get-started/locally/
▶ 数据集定义
https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
▶ 模型各层的编写参考
https://pytorch.org/docs/stable/nn.html

### 1.5 须知
▶ 请勿抄袭；
▶ 请确保代码正确执行，实现图像分类的训练和测试；
▶ 请在群中实名制“学号-姓名”，方便在助教无法执行代码时，
进行后续沟通；
▶ 邮件主题：2024ANN-mid-term-project-学号-姓名
▶ 提交邮箱：sysucsers@163.com
▶ 截止时间：2024-05-19,24:00pm
## 2. 

