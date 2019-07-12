from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

import helper
from data_prep import batch_data
from helper import load_preprocess
from model import TvScriptNet

data_params = Namespace(
    sequence_length=15,
    batch_size=256
)

train_params = Namespace(
    num_epochs=20,
    learning_rate=0.0005,
    vocab_size=21388,  # len(vocab_to_int),
    output_size=21388,  # len(vocab_to_int),
    embedding_dim=256,
    hidden_dim=256,
    n_layers=2,
    show_every_n_batches=300,
)


def forward_back_prop(net, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param net: The PyTorch Module that holds the neural network
    :param optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    # perform backpropagation and optimization
    hidden = tuple([each.data for each in hidden])
    net.zero_grad()
    output, hidden = net(inp, hidden)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden


def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []

    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):

        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)

        for batch_i, (inputs, labels) in enumerate(train_loader, 1):

            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset) // batch_size
            if (batch_i > n_batches):
                break

            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Batch: {} Loss: {}\n'.format(
                    epoch_i, n_epochs, batch_i, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn


if __name__ == '__main__':
    int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
    train_loader = batch_data(int_text, data_params.sequence_length, data_params.batch_size)

    net = TvScriptNet(train_params.vocab_size, train_params.output_size,
                      train_params.embedding_dim, train_params.hidden_dim,
                      train_params.n_layers, dropout=0.5)

    optimizer = torch.optim.Adam(net.parameters(), lr=train_params.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # training the model
    trained_rnn = train_rnn(net, data_params.batch_size, optimizer, criterion,
                            train_params.num_epochs, train_params.show_every_n_batches)

    # saving the trained model
    helper.save_model('save/trained_rnn', trained_rnn)
    print('Model Trained and Saved')
