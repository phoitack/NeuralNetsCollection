import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

# X is input, y is expected output (labels)
X, y = sklearn.datasets.make_moons(200, noise=0.15)

# Plot/print the data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()
print(X.shape, y.shape)

input_neurons = 2
output_neurons = 2
samples = X.shape[0]
learning_rate = 0.01