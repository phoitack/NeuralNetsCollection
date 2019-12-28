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

model_dic = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
def retreive(model_dict):
    W1 = model_dic['W1']
    b1 = model_dic['b1']
    W2 = model_dic['W2']
    b2 = model_dic['b2']
    return W1, b1, W2, b2

# Forward propagation
def forward(x, model_dict):
    W1, b1, W2, b2 = retreive(model_dict)
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    exp_scores = np.exp(a2)
    softmax = exp_scores / np.sum(exp_scores, dim=1, keepdims=True)  #  dim =1 is the columns
    return softmax

# Loss definition
def loss(softmax, y):
    W1, b1, W2, b2 = retreive(model_dict)
    m = np.zeros(200)
    for i, correct_index in enumerate(y):
        predicted = softmax[i][correct_index]
        m[i] = predicted
    log_prob = -np.log(predicted)
    loss = np.sum(log_prob)
    reg_loss = lambda_reg / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    loss += reg_loss
    return float(loss / y.shape[0])

def predict(x, model_dict):
    W1, b1, W2, b2 = retreive(model_dict)
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    exp_scores = np.exp(a2)
    softmax = exp_scores / np.sum(exp_scores, dim=1, keepdims=True)
    return np.argmax(softmax, axis=1)

# Backpropagation


