from string import punctuation
import pickle

import torch

from data_prep import pad_features
from model import SentimentRNN, args_model


def tokenize_review(test_review):
    with open('vocab_to_int.pickle', 'rb') as f:
        vocab_to_int = pickle.load(f)

    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints


def predict(net, test_review, sequence_length=200):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if pred.item() == 1:
        print("Positive review detected!")
    else:
        print("Negative review detected.")


if __name__ == '__main__':
    test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. ' \
                      'This movie had bad acting and the dialogue was slow.'

    net = SentimentRNN(args_model.vocab_size, args_model.output_size,
                       args_model.embedding_dim, args_model.hidden_dim, args_model.n_layers)
    filename = 'models/model_4_1562783667.pth'
    net.load_state_dict(torch.load(filename))

    predict(net, test_review_neg)

