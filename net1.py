import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# X is input, y is expected output (labels)
X, y = sklearn.datasets.make_moons(200, noise=0.15)

# Plot/print the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
print(X.shape, y.shape)

# Net Parameters
input_neurons = 2
output_neurons = 2
samples = X.shape[0]
learning_rate = 0.001
lambda_reg = 0.01

# Dictionary definition
def retreive(model_dict):
    W1 = model_dict['W1']
    b1 = model_dict['b1']
    W2 = model_dict['W2']
    b2 = model_dict['b2']
    return W1, b1, W2, b2


# Forward propagation
def forward(X, model_dict):
    W1, b1, W2, b2 = retreive(model_dict)
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  #  dim =1 is the columns
    return z1, a1, softmax

# Loss definition
def loss(softmax, y, model_dict):
    W1, b1, W2, b2 = retreive(model_dict)
    m = np.zeros(200)
    for i, correct_index in enumerate(y):
        predicted = softmax[i][correct_index]
        m[i] = predicted
    log_prob = -np.log(m)
    loss = np.sum(log_prob)
    reg_loss = lambda_reg / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss += reg_loss
    return float(loss / y.shape[0])


def predict(model_dict, X):
    W1, b1, W2, b2 = retreive(model_dict)
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(softmax, axis=1)


# Backpropagation
def backpropagation(X, y, model_dict, epochs):
    for i in range(epochs):
        W1, b1, W2, b2 = retreive(model_dict)
        z1, a1, probs = forward(X, model_dict)
        delta3 = np.copy(probs)
        delta3[range(X.shape[0]), y] -= 1 # (200,2)  , delta3 = probs -1
        dW2 = (a1.T).dot(delta3)          # a1: (3,200) dot (200,2) --> (3,2)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1-np.power(np.tanh(z1), 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        # L2 regularization
        dW2 += lambda_reg * np.sum(W2)
        dW1 += lambda_reg * np.sum(W1)
        W1 += -learning_rate * dW1
        b1 += -learning_rate * db1
        W2 += -learning_rate * dW2
        b2 += -learning_rate * db2

        # Update model dict.
        model_dict = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Loss print
        if i % 50 == 0:
            print("Loss at epoch {} is: {:.3f}".format(i, loss(probs, y, model_dict)))

    return model_dict


def init_network(input_dim, hidden_dim, output_dim):
    model = {}
    W1 = np.random.randn(input_dim, hidden_dim)/np.sqrt(input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim)/np.sqrt(hidden_dim)
    b2 = np.zeros((1, output_dim))
    model['W1'] = W1
    model['b1'] = b1
    model['W2'] = W2
    model['b2'] = b2
    return model

model_dict = init_network(input_dim = input_neurons, hidden_dim = 3, output_dim = output_neurons)
model = backpropagation(X, y, model_dict, 1500)
