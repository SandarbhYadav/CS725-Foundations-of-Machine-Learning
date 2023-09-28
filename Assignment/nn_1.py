import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# The seed will be fixed to 42 for this assignment.
np.random.seed(42)

NUM_FEATS = 90

feature_dict = {}
for i in range(2, 14):
    feature_dict[i] = "TimbreAvg" + str(i-1)
for j in range(14, 92):
    feature_dict[j] = "TimbreCovariance" + str(j-13)


def relu(Z):
    return np.maximum(0, Z)


def relu_prime(Z):
    return np.where(Z < 0, 0, 1)


class PCA_custom:
    '''
    Custom Principal Component Analysis Class for Feature reduction
    '''

    def __init__(self, principalFeatures=49):
        self.principalFeatures = principalFeatures
        self.features = None
        self._mean = None

    def fit(self, X):
        self._mean = np.mean(X, axis=0)
        X = X - self._mean

        eigenvalues, eigenvector = np.linalg.eig(np.dot(X.T, X) / (X.shape[0] - 1))
        eigenvector = eigenvector.T
        indexs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indexs]
        eigenvector = eigenvector[indexs]

        total = sum(eigenvalues)
        featureVariance = [(i / total) * 100 for i in eigenvalues]

        self.features = eigenvector[:self.principalFeatures]

    def apply(self, X):
        X = X - self._mean
        return np.dot(X, self.features.T)


def PCA_func_custom(X_train_std, X_dev_std, X_test_std, nc):
    # Make an instance of the Model
    pca = PCA_custom(nc)

    # We fit to only our training set
    pca.fit(X_train_std)
    global NUM_FEATS
    NUM_FEATS = pca.principalFeatures

    X_train_proc = pca.apply(X_train_std)
    X_dev_proc = pca.apply(X_dev_std)
    X_test_proc = pca.apply(X_test_std)

    return X_train_proc, X_dev_proc, X_test_proc

class Net(object):
    """
	"""

    def __init__(self, num_layers, num_units):
        """
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also, you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		"""
        self.num_layers = num_layers
        self.num_units = num_units

        self.biases = []
        self.weights = []

        for i in range(num_layers):

            if i == 0:
                # Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
            else:
                # Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))

            self.biases.append(np.random.uniform(-1, 1, size=(1, self.num_units)))

        # Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

        self.nn = {}
        self.op = {}
        self.del_W = {}
        self.del_b = {}

    def __call__(self, X):
        """
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		"""

        self.nn[0] = np.array(X.dot(self.weights[0]) + self.biases[0])
        self.op[0] = np.array(relu(self.nn[0]))
        for i in range(1, self.num_layers + 1):
            self.nn[i] = np.array(self.op[i - 1].dot(self.weights[i]) + self.biases[i])
            if i != self.num_layers:
                self.op[i] = np.array(relu(self.nn[i]))
            else:
                self.op[i] = np.array(self.nn[i])
        return self.op[self.num_layers]

    def backward(self, X, y, lamda):
        """
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		"""
        # Forward pass
        self.__call__(X)

        m = X.shape[0]
        i = self.num_layers
        dLdo = -2 * (y - self.op[i]) / m
        dLdW = self.op[i - 1].T.dot(dLdo)
        dLdb = np.sum(dLdo, axis=0, keepdims=True)
        self.del_W[i] = dLdW + 2 * lamda * self.weights[i]
        self.del_b[i] = dLdb + 2 * lamda * self.biases[i]
        dLdV_dVdU = (dLdo.dot(self.weights[i].T))

        for j in range(self.num_layers - 1, -1, -1):
            dLdn = np.array(dLdV_dVdU, copy=True)
            dLdn = dLdn * relu_prime(self.op[j])
            if j == 0:
                dLdW = X.T.dot(dLdn)
            else:
                dLdW = self.op[j - 1].T.dot(dLdn)
            dLdb = np.sum(dLdn, axis=0, keepdims=True)
            self.del_W[j] = dLdW + 2 * lamda * self.weights[j]
            self.del_b[j] = dLdb + 2 * lamda * self.biases[j]
            dLdV_dVdU = dLdn.dot(self.weights[j].T)

        return self.del_W, self.del_b


class Optimizer(object):
    """
	"""

    def __init__(self, learning_rate, num_layers=2, num_units=78, beta=0.9, gamma=0.81):
        """
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		"""
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.num_units = num_units
        self.beta = beta
        self.gamma = gamma
        self.velocity_weight = []
        self.velocity_bias = []
        self.s_weight = []
        self.s_bias = []
        self.t = 1

        for i in range(num_layers):

            if i == 0:
                # Input layer
                self.velocity_weight.append(np.zeros((NUM_FEATS, self.num_units)))
                self.s_weight.append(np.zeros((NUM_FEATS, self.num_units)))
            else:
                # Hidden layer
                self.velocity_weight.append(np.zeros((self.num_units, self.num_units)))
                self.s_weight.append(np.zeros((self.num_units, self.num_units)))
            self.velocity_bias.append(np.zeros((1, self.num_units)))
            self.s_bias.append(np.zeros((1, self.num_units)))

        # Output layer
        self.velocity_bias.append(np.zeros((1, 1)))
        self.velocity_weight.append(np.zeros((self.num_units, 1)))
        self.s_bias.append(np.zeros((1, 1)))
        self.s_weight.append(np.zeros((self.num_units, 1)))

    def step(self, weights, biases, delta_weights, delta_biases):
        """
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.
		"""
        weight = []
        bias = []
        for i in range(len(weights)):
            self.velocity_weight[i] = self.beta * (self.velocity_weight[i]) + (1 - self.beta) * delta_weights[i]
            self.s_weight[i] = (self.gamma * (self.s_weight[i])) + (
                    (1 - self.gamma) * delta_weights[i] * delta_weights[i])
            self.velocity_weight[i] /= (1 - (self.beta ** self.t))
            self.s_weight[i] /= (1 - (self.gamma ** self.t))
            # weight.append(weights[i] - self.learning_rate * self.velocity_weight[i])
            adap_lr_weight = self.learning_rate / (np.sqrt(self.s_weight[i]) + 1e-7)
            adam_lr_weight = adap_lr_weight * self.velocity_weight[i]
            weight.append(weights[i] - adam_lr_weight)
            self.velocity_bias[i] = self.beta * (self.velocity_bias[i]) + (1 - self.beta) * delta_biases[i]
            self.s_bias[i] = (self.gamma * (self.s_bias[i])) + ((1 - self.gamma) * delta_biases[i] * delta_biases[i])
            self.velocity_bias[i] /= (1 - (self.beta ** self.t))
            self.s_bias[i] /= (1 - (self.gamma ** self.t))
            # bias.append(biases[i] - self.learning_rate * self.velocity_bias[i])
            adap_lr_bias = self.learning_rate / (np.sqrt(self.s_bias[i]) + 1e-7)
            adam_lr_bias = adap_lr_bias * self.velocity_bias[i]
            bias.append(biases[i] - adam_lr_bias)
        self.t += 1
        return weight, bias


def loss_mse(y, y_hat):
    """
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.
	"""
    mse_loss = np.mean(np.square(y - y_hat))
    return mse_loss


def loss_regularization(weights, biases):
    """
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss
	"""
    reg_loss = 0
    for w, b in zip(weights, biases):
        reg_loss += (np.sum(np.square(w)) + np.sum(np.square(b)))
    return reg_loss


def loss_fn(y, y_hat, weights, biases, lamda):
    """
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights: weights of the network
		biases: biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss
	"""
    loss = loss_mse(y, y_hat) + lamda * loss_regularization(weights, biases)
    return loss


def rmse(y, y_hat):
    """
	Compute Root Mean Squared Error (RMSE) loss between ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	"""
    rmse_loss = np.sqrt(np.mean(np.square(y - y_hat)))
    return rmse_loss


def cross_entropy_loss(y, y_hat):
    """
	Compute cross entropy loss

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		cross entropy loss
	"""
    raise NotImplementedError


def train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
):
    """
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each batch of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	"""

    m = train_input.shape[0]
    epoch_loss_old = 0.0
    graph_train = []
    graph_dev = []
    graph = []

    for e in range(max_epochs):
        epoch_loss_new = 0.0

        for i in range(0, m, batch_size):
            batch_input = train_input[i:i + batch_size]
            batch_target = train_target[i:i + batch_size]

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated

            # Compute loss for the batch
            prediction = net(batch_input)
            batch_loss = loss_fn(batch_target, prediction, net.weights, net.biases, lamda)
            epoch_loss_new += batch_loss

        # print(e, i, rmse(batch_target, pred), batch_loss)

        # print(e, epoch_loss_new)

        graph_dev_pred = net(dev_input)
        graph_dev_loss = rmse(dev_target, graph_dev_pred)
        graph_dev.append(graph_dev_loss)

        graph_train_pred = net(train_input)
        graph_train_loss = loss_fn(train_target, graph_train_pred, net.weights, net.biases, lamda)
        graph_train.append(graph_train_loss)

        graph.append(e + 1)

        # Early stopping
        if round(epoch_loss_new, 8) == round(epoch_loss_old, 8):
            break
        else:
            epoch_loss_old = epoch_loss_new
    # Write any early stopping conditions required (only for Part 2)
    # Hint: You can also compute dev_rmse here and use it in the early
    # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
    dev_pred = net(dev_input) + 1922
    dev_loss = rmse(dev_target + 1922, dev_pred)

    # train_pred = net(train_input) + 1922
    # train_loss = rmse(train_target + 1922, train_pred)

    print('MSE on dev set: {:.5f}'.format(dev_loss))
    # print('MSE on train data: {:.5f}'.format(train_loss))

    # y_hat_val = dev_pred
    # pred_df = pd.DataFrame(data=y_hat_val, columns=["Predictions"])
    # pred_df.insert(0, "Id", np.arange(1, len(y_hat_val) + 1, 1.0), True)
    # pred_df["Id"] = pred_df["Id"].astype(int)
    # pred_df.to_csv("22D1594_dev.csv", index=False)

    plt.plot(graph, graph_train, color='r')
    plt.xlabel("Number of epochs")
    plt.ylabel("Regularized MSE loss")
    plt.suptitle("Regularized MSE loss on train set", size=14, fontweight='bold', y=1)
    plt.title("Batch size = "+str(batch_size))
    plt.savefig('train_'+str(batch_size)+'.png')

    plt.clf()

    plt.plot(graph, graph_dev, color='g')
    plt.xlabel("Number of epochs")
    plt.ylabel("RMSE loss")
    plt.suptitle("RMSE loss on dev set", size=14, fontweight='bold', y=1)
    plt.title("Batch size = "+str(batch_size))
    plt.savefig('dev_'+str(batch_size)+'.png')


def get_test_data_predictions(net, inputs):
    """
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	"""
    y_hat = net(inputs)
    y_hat = y_hat + 1922
    pred_df = pd.DataFrame(data=y_hat, columns=["Predictions"])
    pred_df.insert(0, "Id", np.arange(1, len(y_hat) + 1, 1.0), True)
    pred_df["Id"] = pred_df["Id"].astype(int)
    pred_df.to_csv("22D1594.csv", index=False)


def read_data(nc=78):
    """
	Read the train, dev, and test datasets
	"""
    train_data = pd.read_csv("regression/data/train.csv")
    dev_data = pd.read_csv("regression/data/dev.csv")
    test_data = pd.read_csv("regression/data/test.csv")

    train_target = train_data['1']
    train_data.drop('1', axis='columns', inplace=True)
    mean_value = train_data.mean(axis=0)
    std_value = train_data.std(axis=0)
    train_data = (train_data - mean_value) / std_value
    # minimum = train_data.min(axis=0)
    # maximum = train_data.max(axis=0)
    # train_data = (train_data - minimum) / (maximum - minimum)

    dev_target = dev_data['1']
    dev_data.drop('1', axis='columns', inplace=True)
    dev_data = (dev_data - mean_value) / std_value
    # dev_data = (dev_data - minimum) / (maximum - minimum)

    test_data = (test_data - mean_value) / std_value
    # test_data = (test_data - minimum) / (maximum - minimum)

    train_input = train_data.to_numpy()
    dev_input = dev_data.to_numpy()
    test_input = test_data.to_numpy()
    train_target = train_target.to_numpy().reshape((len(train_target), 1))
    dev_target = dev_target.to_numpy().reshape((len(dev_target), 1))

    # Feature Reduction
    train_input, dev_input, test_input = PCA_func_custom(train_input, dev_input, test_input, nc)

    # Target Scaling
    train_target = train_target - min(train_target)
    dev_target = dev_target - min(dev_target)

    return train_input, train_target, dev_input, dev_target, test_input


def main():
    # Hyper-parameters
    max_epochs = 100
    batch_size = 273
    learning_rate = 1e-3
    num_layers = 2
    num_units = 78
    lamda = 188  # Regularization Parameter
    beta = 0.9
    gamma = 0.81
    nc = 78

    train_input, train_target, dev_input, dev_target, test_input = read_data(nc)
    net = Net(num_layers, num_units)
    optimizer = Optimizer(learning_rate, num_layers, num_units, beta, gamma)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )

    #get_test_data_predictions(net, test_input)


if __name__ == '__main__':
    main()
