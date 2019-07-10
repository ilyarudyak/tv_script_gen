from argparse import Namespace
import torch.nn as nn

from data_prep import get_batch

args_model = Namespace(
    vocab_size=74072+1,  # +1 for the 0 padding + our word tokens
    output_size=1,
    embedding_dim=400,
    hidden_dim=256,
    n_layers=2,
    batch_size=50
)


class SentimentRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size(0)

        # embeddings and lstm_out
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        # print(f'lstm_out:{lstm_out.shape}')

        # stack up lstm outputs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # print(f'lstm_out flatten:{lstm_out.shape}')

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        # sigmoid function
        sig_out = self.sig(out)

        # print(f'sig_out:{sig_out.shape}')

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # print(f'sig_out last batch:{sig_out.shape}')

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_batch()
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    net = SentimentRNN(args_model.vocab_size, args_model.output_size,
                       args_model.embedding_dim, args_model.hidden_dim, args_model.n_layers)

    hidden = net.init_hidden(args_model.batch_size)
    sig_out, hidden = net.forward(sample_x, hidden)
    print(sig_out.shape)
