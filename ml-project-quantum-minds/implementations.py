import numpy as np
from helpers import batch_iter, sigmoid


# Function 1: Least Squares
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w_opt: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.
    """
    # Compute the optimal weights using the normal equation
    w_opt = np.linalg.pinv(tx.T.dot(tx)).dot(tx.T).dot(y)

    # Compute the residuals (errors)
    e = y - tx.dot(w_opt)

    # Compute the MSE
    mse = 1 / 2 * np.mean(e**2)

    return w_opt, mse


# Function 2: Logistic Regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D,)
        max_iters: a scalar denoting the number of iterations
        gamma: a scalar denoting the step size

    Returns:
        w: the final weight vector after all iterations
        final_loss: the final loss value after all iterations
    """
    w = initial_w  # Initialize weights

    for _ in range(max_iters):
        # Compute the predicted probabilities
        pred = sigmoid(tx.dot(w))

        # Compute the gradient
        grad = np.dot(tx.T, pred - y) / y.shape[0]

        # Update weights
        w -= gamma * grad

    # Final loss computation with the updated weights
    final_loss = -np.mean(
        y * np.log(sigmoid(tx.dot(w))) + (1 - y) * np.log(1 - sigmoid(tx.dot(w)))
    )

    return w, final_loss


# Function 3: Mean Squared Error using Gradient Descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent to minimize MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, )
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step size

    Returns:
        w: the final weight vector after all iterations
        final_loss: the final loss value (MSE) after all iterations
    """
    w = initial_w

    # Perform gradient descent
    for _ in range(max_iters):
        # Compute the error (residuals)
        err = y - tx.dot(w)

        # Compute the gradient
        grad = -tx.T.dot(err) / len(err)

        # Update weights using the gradient
        w = w - gamma * grad

    # final loss
    final_loss = 1 / 2 * np.mean((y - tx.dot(w)) ** 2)  # MSE

    return w, final_loss


# Function 4: Mean Squared Error using Stochastic Gradient Descent
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent to minimize MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, )
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the step size

    Returns:
        w: the final weight vector after all iterations
        final_loss: the final loss value (MSE) after all iterations
    """
    # Initialize parameters
    w = initial_w

    # Perform stochastic gradient descent
    for _ in range(max_iters):
        for mini_y, mini_tx in batch_iter(
            y, tx, batch_size=1
        ):  # Set batch_size to 1 for SGD
            # Compute the error and the gradient for this mini-batch
            err = mini_y - mini_tx.dot(w)
            stoch_grad = -mini_tx.T.dot(err) / len(err)

            # Update weights using the stochastic gradient
            w = w - gamma * stoch_grad

    # Calculate the final loss on the entire dataset
    err = y - tx.dot(w)
    final_loss = 1 / 2 * np.mean(err**2)  # MSE

    return w, final_loss


# Function 5: Ridge Regression
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, the regularization parameter.

    Returns:
        w_opt: optimal weights, numpy array of shape(D,), D is the number of features.
        mse_loss: scalar, the ridge regression loss (only MSE, without the regularization term).
    """
    N, D = tx.shape

    # Compute the optimal weights using the ridge regression normal equation
    w_opt = np.linalg.solve(tx.T.dot(tx) + 2 * N * lambda_ * np.eye(D), tx.T.dot(y))

    # Compute the residuals (errors)
    e = y - tx.dot(w_opt)

    # Compute the MSE loss
    mse_loss = 1 / (2 * N) * np.sum(e**2)

    return w_opt, mse_loss


# Function 6: Regularized Logistic Regression
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N, D)
        lambda_: scalar, the regularization parameter
        initial_w: numpy array of shape=(D,)
        max_iters: a scalar denoting the number of iterations
        gamma: a scalar denoting the step size

    Returns:
        w: the final weight vector after all iterations
        final_loss: the final loss value after all iterations
    """
    # Initialize weights
    w = initial_w

    for _ in range(max_iters):
        # Compute predictions
        pred = sigmoid(tx.dot(w))

        # Compute the gradient with regularization term
        grad = (tx.T.dot(pred - y) / y.shape[0]) + 2 * lambda_ * w

        # Update weights
        w -= gamma * grad

    # Final loss calculation
    final_pred = sigmoid(tx.dot(w))
    final_loss = -np.mean(y * np.log(final_pred) + (1 - y) * np.log(1 - final_pred))

    return w, final_loss
