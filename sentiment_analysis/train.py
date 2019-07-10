from argparse import Namespace
import torch

from data_prep import get_batch
from model import args_model, SentimentRNN

import numpy as np
import torch.nn as nn

args_train = Namespace(
    lr=0.001,
    epochs=4,
    print_every=100,
    clip=5
)


def train(net, train_loader, valid_loader, optimizer, criterion,
          epochs=args_train.epochs, batch_size=args_model.batch_size,
          print_every=args_train.print_every, clip=args_train.clip):
    counter = 0

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))


if __name__ == '__main__':
    train_loader, valid_loader, test_loader = get_batch()

    net = SentimentRNN(args_model.vocab_size, args_model.output_size,
                       args_model.embedding_dim, args_model.hidden_dim, args_model.n_layers)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args_train.lr)
    train(net, train_loader, valid_loader, optimizer, criterion)
