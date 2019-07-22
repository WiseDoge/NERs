from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        pe.requires_grad = False
        self.pe = pe

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)


class BiLSTMLayer(nn.Module):
    """Bi-LSTM layer.

    This module implements a Bi-LSTM layer.  
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim // 2,
                              num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        # x.shape=(batch_size, max_seq_len, input_dim)
        return self.bilstm(x)


class CNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim,
                              out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # x.shape=(batch_size, max_seq_len, input_dim)
        # cnn_in.shape=(batch_size, input_dim, max_seq_len)
        cnn_in = x.permute(0, 2, 1)
        # self.conv(cnn_in).shape=(batch_size, hidden_dim, max_seq_len)
        # cnn_out.shape=(batch_size, max_seq_len, hidden_dim)
        cnn_out = self.conv(cnn_in).permute(0, 2, 1)
        return F.relu(cnn_out)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: Optional[float] = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: Optional[int] = 4, dropout: Optional[float] = 0.2):
        super().__init__()
        self.num_att_heads = num_heads
        self.att_head_size = hidden_dim // num_heads
        self.all_head_size = hidden_dim

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        # x.shape=(batch_size, max_seq_len, all_head_size<hidden_dim>)
        # new_x_shape=(batch_size, max_seq_len, num_att_heads, att_head_size)
        new_x_shape = x.size()[:-1] + (self.num_att_heads, self.att_head_size)
        x = x.view(*new_x_shape)
        # return shape=(batch_size, num_att_heads, max_seq_len, att_head_size)
        return x.permute(0, 2, 1, 3)

    def extend_attention_mask(self, attention_mask):
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
        return extended_attention_mask

    def forward(self, hidden_states, attention_mask):
        # hidden_states.shape=(batch_size, max_seq_len, hidden_dim)
        # attention_mask.shape=(batch_size, max_seq_len)
        attention_mask = self.extend_attention_mask(attention_mask)
        # attention_mask.shape=(batch_size, 1, 1, max_seq_len)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # query_layer.shape=(batch_size, num_att_heads, max_seq_len, att_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # key_layer.transpose(-1, -2).shape=(batch_size, num_att_heads, att_head_size, max_seq_len) 
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.att_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # attention_scores.shape==(batch_size, num_att_heads, max_seq_len, max_seq_len)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs.shape==(batch_size, num_att_heads, max_seq_len, max_seq_len)
        attention_probs = self.dropout(attention_probs)

        # context_layer.shape=(batch_size, num_att_heads, max_seq_len, att_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer.shape=(batch_size, max_seq_len, num_att_heads, att_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # new_context_layer_shape=(batch_size, max_seq_len, all_head_size)
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CRFLayer(nn.Module):
    # class CRFLayer(torch.jit.ScriptModule):

    """Conditional random field(CRF) layer.

    This module implements a conditional random field by pytorch.
    """

    def __init__(self, tag_size: int):
        super().__init__()
        self.tag_size = tag_size
        self.start_trans = nn.Parameter(torch.empty(tag_size))
        self.end_trans = nn.Parameter(torch.empty(tag_size))
        self.trans = nn.Parameter(torch.empty(tag_size, tag_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)
        nn.init.uniform_(self.trans, -0.1, 0.1)

    def forward(self, emis, tags, mask: Optional[torch.ByteTensor] = None, reduction: str = 'sum'):
        """Compute the NLLLoss.
        
        Args:
            emis: shape=(batch_size, max_seq_len, num_tags).
            tags: shape=(batch_size, max_seq_len).
            mask: shape=(batch_size, max_seq_len).

        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        # shape: (batch_size,)
        numerator = self.compute_non_norm_score(emis, tags, mask)
        # shape: (batch_size,)
        denominator = self.compute_normalizer(emis, mask)
        # shape: (batch_size,)
        nll = denominator - numerator
        return nll.sum()
        # if reduction == 'none':
        #     return nll
        # if reduction == 'sum':
        #     return nll.sum()
        # if reduction == 'mean':
        #     return nll.mean()

    def decode(self, emis, mask) -> List[List[int]]:
        """Decode seqs, only use in test time.
        
        Args:
            emis: shape=(batch_size, max_seq_len, num_tags).
            mask: shape=(batch_size, max_seq_len).
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        if mask is None:
            mask = emis.new_ones(emis.shape[:2], dtype=torch.uint8)
        return self._viterbi_decode(emis, mask)

    def compute_non_norm_score(self, emis, tags, mask):
        """Calculate the non-normalized probability.
        Args:
            emis: shape=(batch_size, max_seq_len, num_tags).
            tags: shape=(batch_size, max_seq_len).
            mask: shape=(batch_size, max_seq_len).
        """
        batch_size, seq_length = tags.shape
        mask = mask.float()
        tags = tags.long()

        # shape = (batch_size, )

        score = self.start_trans[tags[:, 0]] + \
                emis[torch.arange(batch_size), 0, tags[:, 0]]
        # print(score.shape)
        for i in range(1, seq_length):
            trans_score = self.trans[tags[:, i - 1], tags[:, i]] * mask[:, i]
            emiss_score = emis[torch.arange(
                batch_size), i, tags[:, i]] * mask[:, i]

            score += (trans_score + emiss_score)
        seq_lens = mask.long().sum(dim=1)
        last_tags = tags[torch.arange(batch_size), seq_lens - 1]

        score += self.end_trans[last_tags]

        return score

    def compute_normalizer(self, emis, mask):
        """Calculate normalized probability.
        Args:
            emis: shape=(batch_size, max_seq_len, num_tags).
            mask: shape=(batch_size, max_seq_len).
        """
        seq_length = emis.shape[1]
        # mask = mask.byte()

        # self.start_transitions.shape = (num_tags, )
        # emis[:, i, :].shape = (batch_size, num_tags)
        # score.shape = (batch_size, num_tags)
        score = self.start_trans + emis[:, 0, :]

        for i in range(1, seq_length):
            # shape = (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # shape = (batch_size, 1, num_tags)
            broadcast_emis = emis[:, i, :].unsqueeze(1)

            # shape = (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.trans + broadcast_emis

            # shape=(batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # shape = (batch_size, num_tags)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)

        score += self.end_trans
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emis, mask) -> List[List[int]]:
        """Viterbi decode
        Args:
            emis: shape=(batch_size, max_seq_len, num_tags).
            mask: shape=(batch_size, max_seq_len).
        """
        batch_size, seq_length = mask.shape

        # shape = (batch_size, num_tags)
        score = self.start_trans + emis[:, 0, :]

        history = []

        for i in range(1, seq_length):
            # shape = (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # shape = (batch_size, 1, num_tags)
            broadcast_emission = emis[:, i, :].unsqueeze(1)

            # shape = (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.trans + broadcast_emission

            # shape = (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # shape: (batch_size, num_tags)
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)
        score += self.end_trans

        # shape = (batch_size,)
        seq_ends = mask.long().sum(dim=1) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
