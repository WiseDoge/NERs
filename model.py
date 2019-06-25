from layer import *

class BiLSTM(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        return self.hidden2tag(lstm_out)


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.crf = CRFLayer(tag_dim)
        #self.crf = CRF(tag_dim, True)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim
    
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
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.att = SelfAttentionLayer(hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim
        self.norm = LayerNorm(hidden_dim)

    def forward(self, x, attention_mask):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        normout = self.norm(lstm_out + self.att(lstm_out, extended_attention_mask))
        return self.hidden2tag(normout)


class CNN(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim

    def forward(self, x):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        return self.hidden2tag(cnn_out)


class BiLSTMCNN(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = BiLSTMLayer(embed_dim, hidden_dim)
        self.conv = CNNLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        cnn_out = self.conv(lstm_out)
        return self.hidden2tag(cnn_out)


class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.bilstm = BiLSTMLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.tag_dim = tag_dim

    def forward(self, x):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        lstm_out, _ = self.bilstm(cnn_out)
        return self.hidden2tag(lstm_out)


class CNNBiLSTMAtt(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, hidden_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = CNNLayer(embed_dim, hidden_dim)
        self.bilstm = BiLSTMLayer(hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tag_dim)
        self.att = SelfAttentionLayer(hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.tag_dim = tag_dim

    def forward(self, x, attention_mask):
        embeds = self.embedding(x)
        cnn_out = self.conv(embeds)
        lstm_out, _ = self.bilstm(cnn_out)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        normout = self.norm(lstm_out + self.att(lstm_out, extended_attention_mask))
        return self.hidden2tag(normout)


class LogisticRegression(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int, tag_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.hidden2tag = nn.Linear(embed_dim, tag_dim)
        self.tag_dim = tag_dim

    def forward(self, x):
        embeds = self.embedding(x)
        return self.hidden2tag(embeds)

