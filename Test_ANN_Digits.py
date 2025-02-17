from Train_ANN_Digits import *
from utilities import MnistDataloader
import tqdm
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import random




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


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    C = len(parametres) // 2

    Af = activations['A' + str(C)]
    return np.argmax(Af, axis=0)  


def display_images(X, y_true, y_pred, num_images=10, num_columns=5):
    # Calculate the number of rows required with integer division
    num_rows = num_images // num_columns  # Calculate full rows
    if num_images % num_columns != 0:  # If there are remaining images, add an extra row
        num_rows += 1
    
    # Create figure with dynamic size based on number of rows and columns
    plt.figure(figsize=(num_columns * 2, num_rows * 2))
    
    for i in range(num_images):
        # Pick random indices
        index = random.randint(0, X.shape[1] - 1)
        
        # Reshape image from flattened to 28x28
        image = X[:, index].reshape(28, 28)
        
        # True label
        true_label = np.argmax(y_true[:, index])
        
        # Predicted label
        predicted_label = y_pred[index]
        
        # Determine the color: green if correct, red if incorrect
        color = 'green' if true_label == predicted_label else 'red'
        
        # Plot image in the dynamically calculated grid
        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {true_label}, Pred: {predicted_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Dessiner une image de 28x28 pixels
class DrawDigits:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = self.fig.canvas
        self.ax.set_title('Dessinez un chiffre ici')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.imshow(np.ones((28, 28)), cmap='gray', vmin=0, vmax=1)
        self.pixels = np.ones((28, 28))
        
        # Ajouter un bouton pour prédire
        ax_predire = plt.axes([0.75, 0.05, 0.1, 0.075])
        self.btn_predire = Button(ax_predire, 'Prédire')
        self.btn_predire.on_clicked(self.on_predire)

        # Event de clic pour dessiner
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_click)

        self.drawing = False  # Flag to check if mouse is clicked

    def on_mouse_click(self, event):
        if event.inaxes == self.ax:
            self.drawing = not self.drawing  # Toggle drawing

    def on_mouse_move(self, event):
        if event.inaxes == self.ax and self.drawing:
            x, y = int(event.xdata), int(event.ydata)
            self.pixels[y-1:y+2, x-1:x+2] = 0  # Dessine un petit cercle
            self.ax.imshow(self.pixels, cmap='gray', vmin=0, vmax=1)
            self.canvas.draw()

    def on_predire(self, event):
        # Redimensionner l'image à (28x28) et normaliser
        image = self.pixels.flatten().reshape(28, 28) / 1.0
        image = image.T.flatten().reshape(28*28, 1)
        
        # Appel du modèle pour la prédiction
        y_pred = predict(image, parametres)
        
        # Met à jour le titre avec le résultat
        self.ax.set_title(f'Prédiction: {y_pred}')
        plt.draw()  # Met à jour la fenêtre existante

# Fonction principale pour dessiner et prédire
def dessiner_et_predire():
    draw_interface = DrawDigits()
    plt.show()

# Appel pour démarrer l'interface de dessin



def main (X, y, hidden_layers, parametres):
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])

    # dessiner_et_predire()

    y_pred = predict(X, parametres)

    y_test_labels = np.argmax(y_test, axis=0)
    print("Final test accuracy:", accuracy_score(y_test_labels, y_pred))

    display_images(X, y, y_pred, num_images=80, num_columns=10)





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

#print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1) / x_train.max()  # / X_train.max() pour normaliser les données et éviter l'overflow

x_train = x_train.T

x_test = x_test.reshape(x_test.shape[0], -1) / x_train.max()      # / X_train.max() pour normaliser les données et éviter l'overflow
x_test = x_test.T

y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)


m_train = 4000
m_test = 1000
x_train = x_train[:, :m_train]
x_test = x_test[:, :m_test]
y_train = y_train[:, :m_train]
y_test = y_test[:, :m_test]



parametres = deep_neural_network(x_train, y_train, hidden_layers = (256,64), learning_rate = 0.01, n_iter = 1000, n_donnes = 1000)


main(x_test, y_test, (256, 64), parametres)
