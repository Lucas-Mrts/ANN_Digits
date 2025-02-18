from utilities import MnistDataloader
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import random



def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stabilité numérique
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)


def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)].T


def initialisation(dimensions):
    parametres = {}
    C = len(dimensions)

    for c in range(1, C):
        # He initialization for ReLU
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1]) * np.sqrt(2. / dimensions[c - 1])
        parametres['b' + str(c)] = np.zeros((dimensions[c], 1))  # biases initialized to 0

    return parametres


def relu(Z):
    return np.maximum(0, Z)


def relu_derivative(Z):
    return Z > 0


def forward_propagation(X, parametres):
    activations = {'A0': X}
    C = len(parametres) // 2

    for c in range(1, C + 1):
        Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
        if c == C:
            activations['A' + str(c)] = softmax(Z)  # Use softmax for output layer
        else:
            activations['A' + str(c)] = relu(Z)  # Use ReLU for hidden layers

    return activations


def back_propagation(X, y, parametres, activations):
    m = y.shape[1]
    C = len(parametres) // 2
    gradients = {}

    dZ = activations['A' + str(C)] - y  # Derivative of softmax with cross-entropy loss

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1 / m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dA_prev = np.dot(parametres['W' + str(c)].T, dZ)
            dZ = dA_prev * relu_derivative(activations['A' + str(c - 1)])  # Derivative for ReLU

    return gradients


def update(gradients, parametres, learning_rate):
    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] -= learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] -= learning_rate * gradients['db' + str(c)]

    return parametres


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2

    Af = activations['A' + str(C)]
    return np.argmax(Af, axis=0)  # Classe avec la probabilité la plus élevée


def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers=(16, 16), learning_rate=0.01, n_iter=5000, n_donnes=1000):
    
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)
    print("dim :" + str(dimensions))
    
    training_history = np.zeros((int(n_donnes), 2))
    testing_history = np.zeros((int(n_donnes), 2))

    C = len(parametres) // 2

    for i in tqdm.tqdm(range(n_iter)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(X_train, y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        if i % (n_iter // n_donnes) == 0:
            index = i // (n_iter // n_donnes)

            # Convert one-hot encoded labels back to single class labels
            y_train_labels = np.argmax(y_train, axis=0)
            y_test_labels = np.argmax(y_test, axis=0)

            # Train
            training_history[index, 0] = log_loss(y_train.T, Af.T)
            y_pred = predict(X_train, parametres)
            
            training_history[index, 1] = (accuracy_score(y_train_labels, y_pred))
            
            # Test
            activations_test = forward_propagation(X_test, parametres)
            Af_test = activations_test['A' + str(C)]
            testing_history[index, 0] = log_loss(y_test.T, Af_test.T)
            y_pred_test = predict(X_test, parametres)
            testing_history[index, 1] = (accuracy_score(y_test_labels, y_pred_test))


    # Visualisation des résultats
    plt.style.use('dark_background')
    plt.rcParams.update({
        "figure.facecolor":  (0.12 , 0.12, 0.12, 1),
        "axes.facecolor": (0.12 , 0.12, 0.12, 1),
    })

    if(i==n_iter-1):
        print("Final train accuracy:", accuracy_score(y_train_labels, y_pred))
        print("Final test accuracy:", accuracy_score(y_test_labels, y_pred_test))

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

    # Plot loss
    ax1.plot(training_history[:, 0], label='train loss')
    ax1.plot(testing_history[:, 0], label='test loss')
    ax1.set_title('Training and Testing Loss')
    ax1.legend()

    # Plot accuracy
    ax2.plot(training_history[:, 1], label='train acc')
    ax2.plot(testing_history[:, 1], label='test acc')
    ax2.set_title('Training and Testing Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    return parametres



from os.path  import join

input_path = './datasets/mnist/'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath , test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = x_train.reshape(x_train.shape[0], -1) / x_train.max()  # / X_train.max() pour normaliser les données et éviter l'overflow

x_train = x_train.T

x_test = x_test.reshape(x_test.shape[0], -1) / x_train.max()      # / X_train.max() pour normaliser les données et éviter l'overflow
x_test = x_test.T

"""
y_train = y_train.reshape((1, y_train.shape[0]))
y_test = y_test.reshape((1, y_test.shape[0]))
"""

y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)


"""
print("x_train" + str(x_train.shape))
print("y_train" + str(y_train.shape))
print("x_test" + str(x_test.shape))
print("y_test" + str(y_test.shape))
"""

m_train = 2000
m_test = 1000

x_train = x_train[:, :m_train]
x_test = x_test[:, :m_test] 
y_train = y_train[:, :m_train]
y_test = y_test[:, :m_test]


parametres = deep_neural_network(x_train, y_train, x_test, y_test, hidden_layers = (256,64), learning_rate = 0.01, n_iter = 1000, n_donnes = 1000)