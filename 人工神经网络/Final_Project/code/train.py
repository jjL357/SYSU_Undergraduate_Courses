import json
import re
import nltk
import jieba
from collections import Counter
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from tqdm import tqdm
import logging
import os
from pre_process import pre_process
from pre_process import Lang
from model import EncoderRNN, AttnDecoderRNN


# 设置日志记录配置
logging.basicConfig(filename='training.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1
MAX_LENGTH = 50 # 最长的token长度


# sentence to index
def indexesFromSentence_en(lang, sentence):
    return [lang.word2index[word] for word in nltk.word_tokenize(sentence) if word in lang.word2index]
def indexesFromSentence_zh(lang, sentence):
    return [lang.word2index[word] for word in list(jieba.cut(sentence)) if word in lang.word2index]

# 加载数据集
def get_dataloader(data, batch_size, input_lang, output_lang):
    pairs = [(item['en'], item['zh']) for item in data]

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence_en(input_lang, inp)
        tgt_ids = indexesFromSentence_zh(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    dataset = TensorDataset(torch.LongTensor(input_ids).to(device),
                            torch.LongTensor(target_ids).to(device))

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

# 1个epoch的训练
def train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer,criterion):
    total_loss = 0
    for data in tqdm(dataloader):
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        # teaching forcing
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        # free running
        # decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, None)
    
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 计算bleu指标
def evaluate_bleu(dataloader, encoder, decoder, input_lang, output_lang):
    total_bleu = 0
    n = 0
    x = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden,None)

        for i in range(target_tensor.size(0)):
            target = target_tensor[i].cpu().numpy()
            decoded = decoder_outputs[i].argmax(1).cpu().numpy()

            target_sentence = [output_lang.index2word[token] for token in target if token not in (SOS_token, EOS_token) and token in output_lang.index2word]
            decoded_sentence = [output_lang.index2word[token] for token in decoded if token not in (SOS_token, EOS_token) and token in output_lang.index2word]
            if x < 10:
                logging.info(f"target: {target_sentence} decoded: {decoded_sentence}")
                x += 1
            if len(target_sentence) == 0 or len(decoded_sentence) == 0:
                continue
            smooth = SmoothingFunction()
            bleu_score = sentence_bleu([target_sentence], decoded_sentence, smoothing_function=smooth.method1)

            total_bleu += bleu_score
            n += 1
    return total_bleu / n if n > 0 else 0

# 计算beam_search的bleu指标
def evaluate_bleu_beam_search(dataloader, encoder, decoder, input_lang, output_lang, beam_width=3):
    total_bleu = 0
    n = 0
    x = 0
    for data in dataloader:
        input_tensor, target_tensor = data
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        
        decoded_sequences = []
        for i in range(input_tensor.size(0)):
            decoded_sequence = decoder.beam_search(encoder_outputs[i], encoder_hidden[:, i, :], beam_width=beam_width)
            decoded_sequences.append(decoded_sequence)
        
        for i in range(target_tensor.size(0)):
            target = target_tensor[i].cpu().numpy()
            decoded = decoded_sequences[i]

            target_sentence = [output_lang.index2word[token] for token in target if token not in (SOS_token, EOS_token) and token in output_lang.index2word]
            decoded_sentence = [output_lang.index2word[token] for token in decoded if token not in (SOS_token, EOS_token) and token in output_lang.index2word]
            if x < 10:
                logging.info(f"target: {target_sentence} decoded: {decoded_sentence}")
                x += 1
            if len(target_sentence) == 0 or len(decoded_sentence) == 0:
                continue
            smooth = SmoothingFunction()
            bleu_score = sentence_bleu([target_sentence], decoded_sentence, smoothing_function=smooth.method1)

            total_bleu += bleu_score
            n += 1
    return total_bleu / n if n > 0 else 0


import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

# 保存checkpoint
def save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
    }, path)

# 加载checkpoint
def load_checkpoint(path, encoder, decoder, encoder_optimizer, decoder_optimizer):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer_state_dict'])
    return checkpoint['epoch']

def train(train_dataloader, valid_dataloader, encoder, decoder, n_epochs, input_lang, output_lang, learning_rate=0.001, print_every=1, plot_every=1, checkpoint_path='checkpoint.pth', best_checkpoint_path='best_checkpoint.pth'):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    start_epoch = 1
    best_valid_bleu = 0.0

    #if os.path.exists(checkpoint_path):
    #    start_epoch = load_checkpoint(checkpoint_path, encoder, decoder, encoder_optimizer, decoder_optimizer) + 1

    for epoch in range(start_epoch, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            encoder.eval()
            decoder.eval()

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            log_message = '%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
            logging.info(log_message)

            train_bleu = evaluate_bleu(train_dataloader, encoder, decoder, input_lang, output_lang)
            valid_bleu = evaluate_bleu(valid_dataloader, encoder, decoder, input_lang, output_lang)

            logging.info(f'Training BLEU score after epoch {epoch}: {train_bleu:.4f}')
            logging.info(f'Validation BLEU score after epoch {epoch}: {valid_bleu:.4f}')

            if valid_bleu > best_valid_bleu:
                best_valid_bleu = valid_bleu
                save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, best_checkpoint_path)

            encoder.train()
            decoder.train()

            save_checkpoint(encoder, decoder, encoder_optimizer, decoder_optimizer, epoch, checkpoint_path)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        decoder.teaching_rate *= decoder.teaching_decay_rate 
    showPlot(plot_losses)
    
def visualize_Attention(sentence, output_words, attn_weights):

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['SimHei']

    attn_weights = attn_weights.detach().cpu().numpy()
    
    # 检查并调整维度
    if attn_weights.ndim == 1:
        attn_weights = attn_weights.reshape(1, -1)
    elif attn_weights.ndim == 3 and attn_weights.shape[0] == 1:
        attn_weights = attn_weights.squeeze(0)
    
    cax = ax.matshow(attn_weights, cmap='bone')
    fig.colorbar(cax)

    ax.set_xticks(range(len(sentence)))
    ax.set_xticklabels(sentence, rotation=90, fontsize=12)  # 设置字体大小为12
    ax.set_yticks(range(len(output_words)))
    ax.set_yticklabels(output_words, fontsize=12)  # 设置字体大小为12

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # 找到有字的矩形范围并裁剪图像
    plt.tight_layout()
    ax.set_xlim(-0.5, len(sentence) - 0.5)
    ax.set_ylim(len(output_words) - 0.5, -0.5)

    # 保存图像时只保存有字的部分
    fig.savefig("attention.jpg", bbox_inches='tight')

# 训练和推理函数
def main():
    # 数据预处理
    train_num = "10k"
    train_path = 'data/ANN2024_final_translation_dataset_zh_en/train_' + train_num + '.jsonl'
    valid_path = 'data/ANN2024_final_translation_dataset_zh_en/valid.jsonl'
    test_path = 'data/ANN2024_final_translation_dataset_zh_en/test.jsonl'
    train_data, en_lang, zh_lang = pre_process(train_path)
    valid_data, _, _ = pre_process(valid_path)
    test_data, _, _ = pre_process(test_path)
    input_lang, output_lang = en_lang, zh_lang

    # 设置隐藏层大小和批处理大小
    hidden_size = 512
    batch_size = 32

    # 加载数据集
    logging.info("Starting data loader preparation")
    train_dataloader = get_dataloader(train_data, batch_size, input_lang, output_lang)
    valid_dataloader = get_dataloader(valid_data, batch_size, input_lang, output_lang)
    test_dataloader = get_dataloader(test_data, batch_size, input_lang, output_lang)
    logging.info("Data loader preparation done")

    # 初始化encoder和decoder
    logging.info("Initializing models")
    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    #  teaching_rate = 1,teaching_decay_rate = 1 使用teaching forcing
    #  teaching_rate = 0 使用free running
    #  teaching_rate = 1,0 < teaching_decay_rate < 1,先使用teaching forcing 再使用 free running
    
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,teaching_rate = 1,teaching_decay_rate = 0.95).to(device)
    logging.info("Models initialized")

    # 训练优化 
    logging.info("Starting training")
    train(train_dataloader, valid_dataloader, encoder, decoder, 100, input_lang, output_lang, print_every=1, plot_every=5, checkpoint_path='checkpoint.pth', best_checkpoint_path='best_checkpoint.pth')
    logging.info("Training completed")

    # 用best_check_point在测试集测试
    logging.info("Loading best model for evaluation")
    load_checkpoint('best_checkpoint.pth', encoder, decoder, optim.Adam(encoder.parameters()), optim.Adam(decoder.parameters()))
    test_bleu = evaluate_bleu(test_dataloader, encoder, decoder, input_lang, output_lang)

    logging.info(f'Test BLEU score: {test_bleu:.4f}')

if __name__ == "__main__":
    main()
