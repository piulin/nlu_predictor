
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bidirectional,
                 dropout,
                 padding_idx,
                 freeze_embeddings=False):

        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.freeze_embeddings = freeze_embeddings

        if freeze_embeddings:
            print("Freezing word embeddings.")
            self.embedding.requires_grad_(False)


        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, input, lengths , hidden ):


        embedded = self.embedding(input)

        X = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)


        X, hidden = self.lstm(X, hidden)
        # hidden = tuple(self.dropout_layer(hp) for hp in hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        return X, hidden

    def initHidden(self, device, batch_size):
        if self.bidirectional:
            return (
            torch.zeros(2, batch_size, self.hidden_size, device=device), torch.zeros(2, batch_size, self.hidden_size, device=device))
        else:
            return (
            torch.zeros(1, batch_size, self.hidden_size, device=device), torch.zeros(1, batch_size, self.hidden_size, device=device))

    def get_hidden(self, hidden):
        return hidden[0]




class DecoderLSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 output_size,
                 bidirectional,
                 dropout,
                 padding_idx):
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.out = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, input, lengths, hidden):

        X = self.embedding(input)

        X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)

        # output = F.relu(output)
        X, hidden = self.lstm(X, hidden)

        X, _ = torch.nn.utils.rnn.pad_packed_sequence(X, batch_first=True)

        # hidden = tuple(self.dropout_layer(hp) for hp in hidden)
        X = self.out(X)

        # print(f'X: {X}')

        X = self.softmax(X)

        # print(torch. sum(X,dim=2))
        return X, hidden

    def initHidden(self,device, batch_size):
        if self.bidirectional:
            return (torch.zeros(2, batch_size, self.hidden_size, device=device),torch.zeros(2, batch_size, self.hidden_size, device=device))
        else:
            return (torch.zeros(1, batch_size, self.hidden_size, device=device),torch.zeros(1, batch_size, self.hidden_size, device=device))