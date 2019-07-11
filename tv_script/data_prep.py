from helper import load_preprocess
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import torch.nn as nn

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    int_to_vocab = dict(enumerate(set(text)))
    vocab_to_int = {v: k for k, v in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    token_lookup_dict = {
        '.': '<PERIOD>',
        ',': '<COMMA>',
        '"': '<QUOTATION_MARK>',
        ';': '<SEMICOLON>',
        '!': '<EXCLAMATION_MARK>',
        '?': '<QUESTION_MARK>',
        '(': '<LEFT_PARENTHESIS>',
        ')': '<RIGHT_PARENTHESIS>',
        '-': '<DASH>',
        '\n':'<RETURN>'
    }
    return token_lookup_dict


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    features, targets = [], []
    for i in range(len(words)):
        if i+sequence_length < len(words):
            features.append(words[i:i+sequence_length])
            targets.append(words[i+sequence_length])
    data = TensorDataset(torch.from_numpy(np.array(features)),
                         torch.from_numpy(np.array(targets)))
    dataloader = DataLoader(data, shuffle=True, batch_size=batch_size)
#     print(features)
    return dataloader


if __name__ == '__main__':
    # data_dir = 'data/Seinfeld_Scripts.txt'
    # helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
    # int_text, vocab_to_int, int_to_vocab, token_dict = load_preprocess()

    test_text = range(50)
    t_loader = batch_data(test_text, sequence_length=5, batch_size=10)

    data_iter = iter(t_loader)
    sample_x, sample_y = data_iter.next()

    print(sample_x.shape)
    print(sample_x)
    print()
    print(sample_y.shape)
    print(sample_y)

    embedding = nn.Embedding(num_embeddings=50,
                             embedding_dim=25)
    sample_x_embed = embedding(sample_x)
    print(sample_x_embed.shape)











