from string import punctuation
import pickle
from data_prep import get_data


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


if __name__ == '__main__':
    test_review_neg = 'The worst movie I have seen; acting was terrible and I want my money back. ' \
                      'This movie had bad acting and the dialogue was slow.'
    test_ints = tokenize_review(test_review_neg)
    print(test_ints)
