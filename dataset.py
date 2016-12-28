import numpy as np


# ...
# Code is nested in class definition, indentation is not representative.
# "np" stands for numpy.

class Dataset:

    def __init__(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.current_mb = 0
        self.current_batch = 0

    def preprocess(self, input_file):
        with open(input_file, "r") as f:
            data = f.read()

        # count and sort most frequent characters
        chars, occ = np.unique(list(data), return_counts=True)
        self.sorted_chars = chars[np.argsort(-occ)]
        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return np.array(list(map(lambda c: self.char2id[c], list(sequence))))

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return list(map(lambda i: self.id2char[i], encoded_sequence))

    def create_minibatches(self):
        # calculate the number of batches
        self.num_batches = int(self.x.shape[0] / (self.batch_size * self.sequence_length))
        # Is all the data going to be present in the batches? Why?
        # What happens if we select a batch size and sequence length larger than the length of the data?
        assert self.num_batches != 0

        #######################################
        #       Convert data to batches       #
        #######################################


        self.mb = np.array([self.x[i: i + self.sequence_length] for i in range(self.num_batches * self.batch_size + 1)])

    def next_minibatch(self):
        new_epoch = False
        if self.current_batch >= self.num_batches:
            self.current_batch = 0
            new_epoch = True

        var = self.current_batch * self.batch_size
        inp = self.mb[var: var + self.batch_size]
        var += 1
        o = self.mb[var: var + self.batch_size]
        self.current_batch += 1

        batch_x, batch_y = inp, o
        # handling batch pointer & reset
        # new_epoch is a boolean indicating if the batch pointer was reset
        # in this function call
        return new_epoch, batch_x, batch_y

    def one_hot(self, batch, vocab):
        def onehot(x):
            o = np.zeros((x.shape[0], vocab))
            o[np.arange(x.shape[0]), x] = 1
            return o
        if batch.ndim == 1:
            return onehot(batch)
        else:
            return np.array(list(map(onehot, batch)))

def main1():
    dataset = Dataset(1, 40)
    dataset.preprocess("data/selected_conversations.txt")
    dataset.create_minibatches()
    for i in range(4):
        f, s, t = dataset.next_minibatch()
        print(f)
        print("INP")
        print(s)
        print(dataset.decode(s[0]))
        print("OUT")
        print(t)
        print(dataset.decode(t[0]))
        print("\n\n")


def main():
    dat = Dataset(3, 3)
    dat.preprocess("test.txt")
    out = dat.encode("afbdc")
    print(out)
    print(dat.decode(out))
    print(dat.x.shape)
    dat.create_minibatches()
    print("ovo")
    print(dat.mb.shape)
    print(dat.mb)
    print(dat.num_batches)
    for i in range(3):
        f, s, t = dat.next_minibatch()
        print(f)
        print("INP")
        print(s)
        print("OUT")
        print(t)
        print("\n\n")


if __name__ == '__main__':
    main1()