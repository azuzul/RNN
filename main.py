from rnn import RNN as Rnn
import numpy as np
from dataset import Dataset


def run_language_model(dataset, max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100):
    vocab_size = len(dataset.sorted_chars)
    RNN = Rnn(vocab_size, hidden_size, sequence_length, learning_rate)  # initialize the recurrent network

    current_epoch = 0
    batch = 0

    #h0 = np.zeros((hidden_size, dataset.batch_size))
    h0 = np.zeros((dataset.batch_size, hidden_size))
    #TODO
    seed = "HAN:\nIs that good or bad?\n\n" #"Lorem ipsum"#
    n_sample = 300
    average_loss = 0

    while current_epoch < max_epochs:
        e, x, y = dataset.next_minibatch()

        if e:
            current_epoch += 1
            #h0 = np.zeros((hidden_size, dataset.batch_size))
            h0 = np.zeros((dataset.batch_size, hidden_size))
            # why do we reset the hidden state here?

        # One-hot transform the x and y batches
        x_oh, y_oh = dataset.one_hot(x, vocab_size), dataset.one_hot(y, vocab_size)

        # Run the recurrent network on the current batch
        # Since we are using windows of a short length of characters,
        # the step function should return the hidden state at the end
        # of the unroll. You should then use that hidden state as the
        # input for the next minibatch. In this way, we artificially
        # preserve context between batches.
        loss, h0 = RNN.step(h0, x_oh, y_oh)
        average_loss += loss
        if batch % sample_every == 0:
            # run sampling (2.2)
            print("epoch: %d \t batch: %d/%d \t"%(current_epoch, batch%dataset.num_batches, dataset.num_batches), end="")
            print("Average_loss : %f" % (average_loss/(batch*dataset.batch_size)))
            print(sample(RNN, seed, n_sample, dataset))
        batch += 1


def sample(rnn, seed, n_sample, dataset):
    #h0, seed_onehot, samp = np.zeros([rnn.hidden_size, 1]), dataset.one_hot(dataset.encode(seed), rnn.vocab_size), []
    h0, seed_onehot, samp = np.zeros([1, rnn.hidden_size]), dataset.one_hot(dataset.encode(seed), rnn.vocab_size), []
    # inicijalizirati h0 na vektor nula
    # seed string pretvoriti u one-hot reprezentaciju ulaza
    for i in range(n_sample):
        if i >= len(seed):
            h0, _ = rnn.rnn_step_forward(dataset.one_hot(np.array([samp[-1]]), rnn.vocab_size), h0, rnn.U, rnn.W, rnn.b)
        else:
            h0, _ = rnn.rnn_step_forward(seed_onehot[i].reshape([1, -1]), h0, rnn.U, rnn.W, rnn.b)
        samp.append(np.argmax(rnn.output(h0, rnn.V, rnn.c)))
    return dataset.decode(samp)#"".join(dataset.decode(samp))


def main():
    #TODO
    dataset = Dataset(30, 15)
    dataset.preprocess("data/selected_conversations.txt")
    dataset.create_minibatches()
    run_language_model(dataset, 100000, sequence_length=dataset.sequence_length)

if __name__ == '__main__':
    main()
