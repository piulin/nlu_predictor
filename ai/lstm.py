
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, dropout, freeze_embeddings=False):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.freeze_embeddings = freeze_embeddings
        if freeze_embeddings:
            print("Freezing word embeddings.")
            self.embedding.requires_grad_(False)
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, input, hidden, embedded = None):

        if embedded == None:
            embedded = self.embedding(input).view(1, 1, -1)

        output = embedded
        output, hidden = self.lstm(output, hidden)
        hidden = tuple(self.dropout_layer(hp) for hp in hidden)
        return output, hidden

    def initHidden(self, device):
        if self.bidirectional:
            return (
            torch.zeros(2, 1, self.hidden_size, device=device), torch.zeros(2, 1, self.hidden_size, device=device))
        else:
            return (
            torch.zeros(1, 1, self.hidden_size, device=device), torch.zeros(1, 1, self.hidden_size, device=device))

    def get_hidden(self, hidden):
        return hidden[0]




class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, bidirectional, dropout):
        print(hidden_size)
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)
        self.out = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, input, hidden):

        output = self.embedding(input).view(1, 1, -1)
        # output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        hidden = tuple(self.dropout_layer(hp) for hp in hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self,device):
        if self.bidirectional:
            return (torch.zeros(2, 1, self.hidden_size, device=device),torch.zeros(2, 1, self.hidden_size, device=device))
        else:
            return (torch.zeros(1, 1, self.hidden_size, device=device),torch.zeros(1, 1, self.hidden_size, device=device))