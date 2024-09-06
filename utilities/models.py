import torch

import torch.nn as nn


class news_classifier(nn.ModuleList):

    def __init__(self, lstm_layers, hidden_dim, embedding_dim, device, vocab_size, num_classes, batch_size):
        super(news_classifier, self).__init__()
        # setting parameters
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.LSTM_layers = lstm_layers
        self.input_size = vocab_size
        self.device = device

        # defining embedding layer
        self.embedding = nn.Embedding(
            self.input_size, self.embedding_dim, padding_idx=0)

        # defining an multi-LSTM layer
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=self.LSTM_layers, batch_first=True)

        # defining a MLP for classification 
        self.linear = nn.Sequential(
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, x):
        # initializing hidden states
        h = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        h = h.to(self.device)

        c = torch.zeros((self.LSTM_layers, x.size(0), self.hidden_dim))
        c = c.to(self.device)

        # normalizing the hidden states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        # passing the input through each layers
        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.linear(out[:, -1, :])

        return out
