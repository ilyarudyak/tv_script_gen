import numpy as np
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_data():

    with open('data/reviews.txt', 'r') as f:
        reviews = f.read()
    with open('data/labels.txt', 'r') as f:
        labels = f.read()

    # get rid of punctuation
    reviews = reviews.lower()  # lowercase, standardize
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # split by new lines and spaces
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)

    # create a list of words
    words = all_text.split()

    ## Build a dictionary that maps words to integers
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    ## use the dict to tokenize each review in reviews_split
    ## store the tokenized reviews in reviews_ints
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

    # get indices of any reviews with length 0
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    # remove 0-length reviews and their labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

    return reviews_ints, encoded_labels


def pad_features(reviews_ints, seq_length=200):
    """ Return features of review_ints, where each review is padded with 0's
        or truncated to the input seq_length.
    """

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def split_data(split_frac=0.8):
    reviews_ints, encoded_labels = get_data()
    features = pad_features(reviews_ints)

    split_idx = int(len(features) * split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    return train_x, val_x, test_x, train_y, val_y, test_y


def get_batch(batch_size=50):
    train_x, val_x, test_x, train_y, val_y, test_y = split_data()

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # make sure the SHUFFLE your training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_x, val_x, test_x, train_y, val_y, test_y = split_data()

    print(train_x.shape)

    # train_loader, valid_loader, test_loader = get_batch()
    #
    # # obtain one batch of training data
    # dataiter = iter(train_loader)
    # sample_x, sample_y = dataiter.next()
    #
    # print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    # print('Sample input: \n', sample_x)
    # print()
    # print('Sample label size: ', sample_y.size())  # batch_size
    # print('Sample label: \n', sample_y)

