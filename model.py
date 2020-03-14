import torch
import torch.nn as nn

X = torch.tensor(([2, 9], [1, 5], [3, 6]), dtype=torch.float)
Y = torch.tensor(([92, 100, 89]), dtype=torch.int)

x_predicted = torch.tensor(([4, 8]), dtype=torch.float)

X_max, index = torch.max(X, 0)
X = torch.div(X, X_max)

print(X_max)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork).__init__()

        # layer sizes
        self.inputSize = 2
        self.hiddenSize = 3
        self.outputSize = 1

        # weight initialization
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.W2 = torch.randn(self.hiddenSize, self.outputSize)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def d_sigmoid(self, x):
        sig = torch.sigmoid(x)
        return sig * (1 - sig)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1)
        self.a1 = self.sigmoid(self.z1)
        self.z2 = torch.matmul(self.a1)
        self.a2 = self.sigmoid(self.z2)  # output
        return self.a2

    def backward(self, X, y, o):
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o)
        self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(X), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

NN = NeuralNetwork()