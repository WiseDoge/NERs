from .layer import *
import torch 
import torch.nn as nn
import math

class BiLSTM(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.tag_size = tag_size

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        return self.hidden2tag(lstm_out)


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.crf = CRFLayer(tag_size)
        #self.crf = CRF(tag_size, True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.tag_size = tag_size
    
    def forward(self, x, tags, crf_mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        emis = self.hidden2tag(lstm_out)
        return self.crf(emis, tags, crf_mask)
    
    def decode(self, x, crf_mask=None):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        emis = self.hidden2tag(lstm_out)
        return self.crf.decode(emis, crf_mask)


class BiLSTMAtt(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.att = SelfAttentionLayer(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.tag_size = tag_size
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, attention_mask):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        normout = self.norm(lstm_out + self.att(lstm_out, attention_mask))
        return self.hidden2tag(normout)


class CNN(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, max_seq_len, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.embed_dim = embed_dim
        self.tag_size = tag_size

    def forward(self, x):
        embeds = self.embedding(x) * math.sqrt(self.embed_dim)
        embeds = self.pos_encoder(embeds)
        cnn_out = self.conv(embeds)
        return self.hidden2tag(cnn_out)


class BiLSTMCNN(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.conv = CNNLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.tag_size = tag_size

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        cnn_out = self.conv(lstm_out)
        return self.hidden2tag(cnn_out)


class CNNBiLSTM(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.bilstm = BiLSTMLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.tag_size = tag_size

    def forward(self, x):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        lstm_out, _ = self.bilstm(cnn_out)
        return self.hidden2tag(lstm_out)


class CNNBiLSTMAtt(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, hidden_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.bilstm = BiLSTMLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)
        self.att = SelfAttentionLayer(hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.tag_size = tag_size

    def forward(self, x, attention_mask):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        lstm_out, _ = self.bilstm(cnn_out)
        normout = self.norm(lstm_out + self.att(lstm_out, attention_mask))
        return self.hidden2tag(normout)


class LogisticRegression(nn.Module):
    def __init__(self, vocab:int, embed_dim:int, tag_size:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed_dim)
        self.hidden2tag = nn.Linear(embed_dim, tag_size)
        self.tag_size = tag_size

    def forward(self, x):
        embeds = self.embedding(x)
        return self.hidden2tag(embeds)

