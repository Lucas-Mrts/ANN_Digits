# Deep Neural Network for MNIST Classification

This project implements a deep neural network in Python for handwritten digit classification using the MNIST dataset. The implementation includes forward propagation, backpropagation, and weight optimization using gradient descent.

## Usage

Run the following command to start training:

```python
parametres = deep_neural_network(x_train, y_train, x_test, y_test, hidden_layers=(256, 64), learning_rate=0.01, n_iter=1000, n_donnes=1000)
```

### Modifiable Parameters:

- `hidden_layers`: Tuple defining the number and size of hidden layers (e.g., `(128, 64, 32)`).
- `learning_rate`: Learning rate for gradient descent (e.g., `0.01` or `0.001`).
- `n_iter`: Number of training iterations.
- `n_donnes`: Number of data points used for tracking training progress.
- `m_train`, `m_test`: Number of training and test samples used (defined in the dataset loading section).



On `ANN_Digits.py` : at the end of training, the final accuracy for both training and test sets is displayed.

![alt text](image-1.png)

On `Test_ANN_Digits.py` : at the end of training, the final accuracy for both training and test sets is displayed.



## Potential Improvements

- Add noise to the input data to make the model more robust.
- Implement regularization techniques (dropout, L2, batch normalization)





