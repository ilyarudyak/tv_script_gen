from torch import nn


class CharRNN(nn.Module):

    def __init__(self, chars, n_hidden=256, n_layers=2,
                 drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = chars
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ## TODO: define the layers of the model
        self.lstm = nn.LSTM(len(chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.output = nn.Linear(n_hidden, len(chars))

    def forward(self, x, hidden):
        """ Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. """

        ## TODO: Get the outputs and the new hidden state from the lstm
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.contiguous().view(-1, self.n_hidden)
        out = self.output(x)

        # return the final output and the hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden
