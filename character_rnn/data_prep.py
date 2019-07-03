import numpy as np


def one_hot_encode(arr, n_labels):
    # Initialize the the encoded array
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # Fill the appropriate elements with ones
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # Finally reshape it to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def get_batches(arr, batch_size, seq_length):
    """Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    """

    ## TODO: Get the number of batches we can make
    n_batches = int(arr.size / (batch_size * seq_length))

    ## TODO: Keep only enough characters to make full batches
    arr = arr[:(n_batches * batch_size * seq_length)]

    ## TODO: Reshape into batch_size rows
    arr = arr.reshape((batch_size, -1))

    ## TODO: Iterate over the batches using a window of size seq_length
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:(n + seq_length)]
        # The targets, shifted by one
        y = np.zeros_like(x)
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y


def get_encoded():
    with open('data/anna.txt', 'r') as f:
        text = f.read()

    # encode the text and map each character to an integer and vice versa

    # we create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    return chars, encoded


if __name__ == '__main__':
    # test_seq = np.array([[3, 5, 1]])
    # one_hot = one_hot_encode(test_seq, 8)
    #
    # print(one_hot)

    _, encoded = get_encoded()
    batches = get_batches(encoded, 8, 50)
    x, y = next(batches)
    # printing out the first 10 items in a sequence
    print('x\n', x[:10, :10])
    print('\ny\n', y[:10, :10])