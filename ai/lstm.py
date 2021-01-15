
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLSTM(nn.Module):
    def __init__(self,
                 number_of_words,
                 embeddings_size,
                 hidden_size,
                 bidirectional,
                 dropout,
                 padding_idx,
                 freeze_embeddings=False):

        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embeddings_size = embeddings_size
        self.embedding = nn.Embedding(number_of_words, embeddings_size, padding_idx=padding_idx)
        self.freeze_embeddings = freeze_embeddings

        if freeze_embeddings:
            print("Freezing word embeddings.")
            self.embedding.requires_grad_(False)


        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(embeddings_size, hidden_size, bidirectional=bidirectional, batch_first=True)
        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, input, lengths , hidden, sorted=True ):


        embedded = self.embedding(input)

        embedded = self.dropout_layer(embedded)

        X = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=sorted)


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
                 number_of_slots,
                 embeddings_size,
                 hidden_size,
                 bidirectional,
                 dropout,
                 padding_idx):
        super(DecoderLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(number_of_slots, embeddings_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embeddings_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, number_of_slots)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward( self, input, lengths, hidden, sorted=True ):

        X = self.embedding(input)

        X = self.dropout_layer(X)

        X = torch.nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True, enforce_sorted=sorted)

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
            return (torch.zeros(1, batch_size, self.hidden_size, device=device),torch.zeros(1, batch_size, self.hidden_size, device=device))