from argparse import Namespace
import torch
from torch import nn
from data_prep import *
from model import CharRNN
import time, pickle


args = Namespace(
    n_hidden=512,
    n_layers=2,
    batch_size=128,
    seq_length=100,
    n_epochs=20,
    lr=0.001,
    print_every=10
)


def train(model, train_data, epochs=10, batch_size=10, seq_length=50,
          lr=0.001, clip=5, val_frac=0.1, print_every=10):
    """ Training a network

        Arguments
        ---------

        model: CharRNN network
        train_data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    """
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(train_data) * (1 - val_frac))
    train_data, val_data = train_data[:val_idx], train_data[val_idx:]

    counter = 0
    n_chars = len(model.chars)
    train_losses, val_losses = [], []
    for epoch in range(epochs):

        # boiler plate code
        h_train = model.init_hidden(batch_size)

        cur_train_losses = []
        for x_train, y_train in get_tensor_batches(train_data, batch_size, seq_length, n_chars):
            counter += 1

            # boiler plate code
            h_train = tuple([each.data for each in h_train])
            model.zero_grad()

            ################# forward pass #################
            output, h_train = model(x_train, h_train)
            loss = criterion(output, y_train.view(batch_size * seq_length).long())
            loss.backward()
            cur_train_losses.append(loss.item())
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            ################# forward pass #################

            # loss stats
            if counter % print_every == 0:
                print("Epoch: {}/{}...".format(epoch + 1, epochs),
                      "Step: {}...".format(counter))

        # Get validation loss
        h_val = model.init_hidden(batch_size)
        cur_val_losses = []
        model.eval()
        for x_val, y_val in get_tensor_batches(val_data, batch_size, seq_length, n_chars):

            # boiler plate code
            h_val = tuple([each.data for each in h_val])

            ################# forward pass #################
            output_val, h_val = model(x_val, h_val)
            val_loss = criterion(output_val, y_val.view(batch_size * seq_length).long())
            cur_val_losses.append(val_loss.item())
            ################# forward pass #################

        model.train()

        cur_train_loss, cur_val_loss = np.mean(cur_train_losses), np.mean(cur_val_losses)
        train_losses.append(cur_train_loss)
        val_losses.append(cur_val_loss)
        print("Epoch: {}/{}...".format(epoch + 1, epochs),
              "Train Loss: {:.4f}...".format(cur_train_loss),
              "Validation Loss: {:.4f}".format(cur_val_loss))

    save_losses(train_losses, val_losses)


def save_losses(train_losses, val_losses):
    losses_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open('losses_' + str(int(time.time())) + '.pickle', 'wb') as f:
        pickle.dump(losses_dict, f)


def load_losses(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f)
    return d['train_losses'], d['val_losses']


if __name__ == '__main__':
    chars, encoded = get_encoded()
    char_rnn_model = CharRNN(chars, args.n_hidden, args.n_layers)

    train(char_rnn_model, encoded, epochs=args.n_epochs, batch_size=args.batch_size,
          seq_length=args.seq_length, lr=args.lr, print_every=args.print_every)
