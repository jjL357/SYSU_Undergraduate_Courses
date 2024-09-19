import torch
import torch.nn as nn
import torch.nn.functional as F
import random

MAX_LENGTH = 50

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,dropout=dropout_p)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden

class DotProductAttention(nn.Module):
    def __init__(self, hidden_size):
      super(DotProductAttention, self).__init__()
      self.hidden_size = hidden_size

    def forward(self, query, keys):
      scores = torch.bmm(keys, query.transpose(1, 2))  
      scores = scores.squeeze(2) 
      weights = F.softmax(scores, dim=1)  
      context = torch.bmm(weights.unsqueeze(1), keys)  
      return context, weights    

class MultiplicativeAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiplicativeAttention, self).__init__()
        self.hidden_size = hidden_size
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, query, keys):
        transformed_query = self.Wa(query)
        scores = torch.bmm(keys, transformed_query.transpose(1, 2)).squeeze(2) 
        weights = F.softmax(scores, dim=1)  
        context = torch.bmm(weights.unsqueeze(1), keys)  
        return context, weights

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=2, dropout_p=0.1,teaching_rate = 1,teaching_decay_rate = 1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, num_layers=num_layers, batch_first=True,  dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.teaching_rate = teaching_rate # 使用teaching forcing的概率
        self.teaching_decay_rate = teaching_decay_rate # 每次teaching foring :self.teaching_rate *= self.teaching_decay_rate 

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None and random.random() < self.teaching_rate:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden[-1].unsqueeze(0).permute(1, 0, 2)
        # beam_search
        # query = hidden[-1].unsqueeze(1)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

    
    def beam_search(self, encoder_outputs, encoder_hidden, max_length=MAX_LENGTH, beam_width=5):
        k = beam_width
        sequences = [[list(), 0.0, encoder_hidden]]  # list of (sequence, score, hidden)

        for _ in range(max_length):
            all_candidates = list()
            for seq, score, hidden in sequences:
                decoder_input = torch.tensor([seq[-1]]).unsqueeze(0).to(device) if seq else torch.tensor([SOS_token]).unsqueeze(0).to(device)
                decoder_output, hidden, attn_weights = self.forward_step(decoder_input, hidden, encoder_outputs)
                
                topk = decoder_output.topk(k)
                for i in range(k):
                    candidate = [seq + [topk[1][0][i].item()], score - topk[0][0][i].item(), hidden]
                    all_candidates.append(candidate)
                    
            ordered = sorted(all_candidates, key=lambda x: x[1])
            sequences = ordered[:k]
        
        return sequences[0][0]  # return the best sequence

