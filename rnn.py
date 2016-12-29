import numpy as np


class RNN:
    def __init__(self, vocab_size, hidden_size=100, sequence_length=30, learning_rate=1e-1):
        self.hidden_size = hidden_size  # dimenzija skirvenog sloja
        self.sequence_length = sequence_length  # duljina niza znakova nad kojim ucimo, nema veze s matricama
        self.vocab_size = vocab_size  # dimenzija ulaznog sloja
        self.learning_rate = learning_rate  # stopa ucenja

        self.U = 1e-2 * np.random.rand(vocab_size, hidden_size)  # input projection  D X H
        self.W = 1e-2 * np.random.rand(hidden_size, hidden_size)  # hidden-to-hidden projection H x H
        self.b = np.zeros([hidden_size, 1])  # input bias H x 1

        self.V = 1e-2 * np.random.rand(vocab_size, hidden_size)  # output projection
        self.c = np.zeros([vocab_size, 1])  # output bias

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U, self.memory_W, self.memory_V = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

    def rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h_current = np.tanh(np.dot(W, h_prev) + np.dot(U.T, x.T) + b)   # BS x H
        h_current, cache = h_current, (W, x, h_prev, h_current)

        # return the new hidden state and a tuple of values needed for the backward step
        return h_current, cache

    def rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        h, cache = [h0], []
        for t in range(self.sequence_length):
            tmp, c = self.rnn_step_forward(x[:, t, :], h[-1], U, W, b)
            h.append(tmp)
            cache.append(c)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        h = np.transpose(np.array(h[1:]), [1, 0, 2])
        return h, cache

    def rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        W, x, h_prev = cache
        dh_prev, dU, dW = np.dot(grad_next.T, W), np.dot(x.T, grad_next.T), np.dot(grad_next, h_prev.T)
        db = np.sum(grad_next, axis=1).reshape([-1, 1])
        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters
        return dh_prev, dU, dW, db

    def rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU, dW, db = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.b)

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        dp = np.zeros_like(dh[0])
        for t in range(self.sequence_length-1, 0, -1):
            W, x, h_prev, h_curr = cache[t]
            dh_curr = dh[t] + np.dot(self.W, 1 - h_curr**2)  # trenutni grad + upstream * dht/dat * dat/dgt-1
            dp, du, dw, dB = self.rnn_step_backward(dh_curr, (W, x, h_prev))
            dU, db, dW = du + dU, db + dB, dW + dw
        return np.clip(dU, -5, 5), np.clip(dW, -5, 5), np.clip(db, -5, 5)

    def output(self, h, V, c):
        # Calculate the output probabilities of the network
        return np.dot(V, h) + c    # leng x BS

    def output_loss_and_grads(self, h, V, c, y):
        """Calculate the loss of the network for each of the outputs

        h - hidden states of the network for each timestep.
            the dimensionality of h is (batch size x sequence length x hidden size
            (the initial state is irrelevant for the output)
        V - the output projection matrix of dimension hidden size x vocabulary size
        c - the output bias of dimension vocabulary size x 1
        y - the true class distribution - a one-hot vector of dimension
            vocabulary size x 1 - you need to do this conversion prior to
            passing the argument. A fast way to create a one-hot vector from
            an id could be something like the following code:

           y[timestep] = np.zeros((vocabulary_size, 1))
           y[timestep][batch_y[timestep]] = 1

           where y might be a dictionary. """

        loss, dh, dV, dc = 0.0, [], np.zeros_like(self.V), np.zeros_like(self.c)
        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h
        for t in range(self.sequence_length):
            hp = h[:, t, :]  # H x BS
            o = self.output(hp, V, c)  # leng x BS
            exp = np.exp(o)  # leng x BS
            s = exp / np.sum(exp, axis=0, keepdims=True)  # leng x BS
            yp = y[:, t, :].T
            dO = s - yp  # leng x BS
            dV += np.dot(dO, hp.T)  # ( leng x BS ) * ( H x BS ).T = leng x H
            dc += np.sum(dO, axis=1).reshape([-1, 1])  #
            dh.append(np.dot(self.V.T, dO))  # ( leng x H ).T * ( leng x BS ) = ( BS x H )
            loss += -np.sum(np.log(s)*yp)
        return loss, np.array(dh), dV, dc

    # The inputs to the function are just indicative since the variables are mostly present as class properties

    def update(self, dU, dW, db, dV, dc):
        # update memory matrices
        # perform the Adagrad update of parameters
        def comUp(r, g, eps=self.learning_rate):
            return - eps/(1e-7 + np.sqrt(r)) * g
        self.memory_U, self.memory_W, self.memory_b = self.memory_U + dU ** 2, self.memory_W + dW ** 2, self.memory_b + db ** 2
        self.memory_c, self.memory_V = self.memory_c + dc ** 2, self.memory_V + dV ** 2
        upU, upW, upV,  = comUp(self.memory_U, dU), comUp(self.memory_W, dW), comUp(self.memory_V, dV)
        upC, upB = comUp(self.memory_c, dc), comUp(self.memory_b, db)
        self.U, self.W, self.V, self.c, self.b = self.U + upU, self.W + upW, self.V + upV, self.c + upC, self.b + upB

    def step(self, h, x, y):
        h, cache = self.rnn_forward(x, h, self.U, self.W, self.b)
        loss, dh, dV, dc = self.output_loss_and_grads(h, self.V, self.c, y)
        dU, dW, db = self.rnn_backward(dh, cache)
        self.update(dU, dW, db, dV, dc)
        return loss, h[:, -1, :]


