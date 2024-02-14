import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DS_PATH = '../../datasets/digit-recognizer'

def load_dataset():
    train_data = pd.read_csv(DS_PATH + '/train.csv')
    test_data = pd.read_csv(DS_PATH + '/test.csv')

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    np.random.shuffle(train_data)

    val_data = train_data[0:1000].T
    val_y = val_data[0]
    val_X = val_data[1:]/255

    train_data = train_data[1000:].T
    train_y = train_data[0]
    train_X = train_data[1:]/255

    test_data = test_data.T
    test_y = test_data[0]
    test_X = test_data[1:]/255

    return train_X, train_y, val_X, val_y, test_X, test_y

# Activation Functions
# ReLU activation function
# x if x > 0
# 0 if x <= 0
def relu(x):
    return np.maximum(0, x)

def deriv_relu(x):
    return x > 0

# Sigmoid
def sigmoid(x):
    pass

# Softmax
def softmax(x):
    """
        e^(zi) / sum(j=1 -> K, e^(zj))
    """
    return np.exp(x) / (np.sum(np.exp(x)))

def one_hot_encode(x):
    one_hot = np.zeros((x.size, x.max() + 1))
    one_hot[np.arange(x.size), x] = 1
    
    return one_hot.T

def accuracy(pred, y):
    return np.sum(pred == y) / y.size

class Module:
    def __init__(self):
        self.input_tensor = None
        #self.output_tensor = None
        self.m = None

    def __call__(self, x):
        self.input_tensor = x
        x = self.forward(x)
        #self.output_tensor = x

        return x
    
    def forward(self, x):
        return x
    
    def backward(self):
        pass

    def update_params(self):
        pass

class Linear(Module):
    """
        This will represent a dense layer
        Z = weights * A + bias
    """
    def __init__(self, inputs:int, outputs:int, weights_initializer=np.random.standard_normal, bias_initializer=np.random.standard_normal):
        super().__init__()

        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights_initializer((self.outputs, self.inputs))
        self.bias = bias_initializer((self.outputs, 1))
        self.grad_weights = np.zeros((self.outputs, self.inputs))
        self.grad_bias = np.zeros((self.outputs, 1))
        self.grad_out = np.zeros((1, self.inputs))

    def forward(self, x):
        return self.weights.dot(x) + self.bias

    def backward(self):
        """
            d(Z[l]) = weights[l+1].T * d(Z[l+1]) @ g'(Z[l])
            d(W[l]) = (1/m) * d(Z[l]) * x.T
            d(b[l]) = (1/m) * sum(d(Z[l])) 

            For final layer:
                d(Z[l]) = A[l] - Y

            weights = weights - alpha * d(weights)
            bias = bias - alpha * d(bias)
        """
        self.grad_weights = 1 / self.m * self.grad_out.dot(self.input_tensor.T)
        self.grad_bias = 1 / self.m * np.sum(self.grad_out, 1)

    def update_params(self):
        pass
        

class NeuralNet(Module):
    def __init__(self):
        self.l1 = Linear(784, 10)
        self.l2 = Linear(10, 10)

        self.grad_out = None

    def forward(self, x):
        x = self.l1(x)
        x = relu(x)
        x = self.l2(x)
        x = softmax(x)

        self.m = x.size
        self.l1.m = self.m
        self.l2.m = self.m

        return x

    def backward(self):
        self.l2.grad_out = self.grad_out
        self.l2.backward()
        self.l1.grad_out = self.l2.grad_weights.T.dot(self.grad_out) * deriv_relu(self.l2.input_tensor)
        self.l1.backward()

    def update_params(self):
        pass

if __name__ == '__main__':
    train_X, train_y, val_X, val_y, test_X, test_y = load_dataset()

    # Uncomment for testing
    #print(train_X.shape)
    #print(train_X[:, 0])

    train_y = one_hot_encode(train_y)
    val_y = one_hot_encode(val_y)
    test_y = one_hot_encode(test_y)

    iterations = 500
    lr = 1e-1

    model = NeuralNet()
    
    for i in range(iterations):
        pred = model(train_X)

        #loss = train_y - pred
        loss = pred - train_y
        model.grad_out = loss

        model.backward()

        # Gradient descend
        model.l1.weights -= lr * model.l1.grad_weights
        model.l1.bias -= lr * np.reshape(model.l1.grad_bias, (10,1))
        model.l2.weights -= lr * model.l2.grad_weights
        model.l2.bias -= lr * np.reshape(model.l2.grad_bias, (10,1))

        if i % 100 == 0:
            print(f"iteration: {i} - loss: {np.sum(loss)} - acc: {accuracy(np.argmax(pred, axis=0), train_y)}")

    pred = model(val_X)
    #loss = val_y - pred
    loss = pred - val_y

    print(f"val_loss: {np.sum(loss)} - acc: {accuracy(np.argmax(pred, axis=0), val_y)}")

    print(np.argmax(model(val_X[:,3, None]), axis=0))
    print(np.argmax(val_y[3], 0))
    plt.imshow(np.reshape(val_X[:,3, None], (28,28)))
    plt.show()
