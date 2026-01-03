"""
Utility functions for Logistic Regression

This module contains reusable functions for:
- Sigmoid activation
- Cost computation (log loss)
- Gradient computation
- Gradient Descent optimization
- Predictions and evaluation
"""

import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid (logistic) function.
    
    Parameters:
    -----------
    z : numpy.ndarray or float
        Input value(s)
    
    Returns:
    --------
    sigmoid : numpy.ndarray or float
        Sigmoid of input, bounded between 0 and 1
    """
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    """
    Compute the log loss (binary cross-entropy) cost function.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix of shape (m, n+1)including bias column
    y : numpy.ndarray
        Target values of shape (m, 1), values in {0, 1}
    theta : numpy.ndarray
        Parameters of shape (n+1, 1)
    
    Returns:
    --------
    cost : float
        The log loss
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # Avoid log(0) by clipping predictions
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost


def compute_gradient(X, y, theta):
    """
    Compute the gradient of the cost function.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix of shape (m, n+1)
    y : numpy.ndarray
        Target values of shape (m, 1)
    theta : numpy.ndarray
        Parameters of shape (n+1, 1)
    
    Returns:
    --------
    gradients : numpy.ndarray
        Gradients of shape (n+1, 1)
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    gradients = (1/m) * X.T.dot(h - y)
    return gradients


def gradient_descent(X, y, theta=None, learning_rate=0.01, n_iterations=1000,
                     tolerance=1e-6, verbose=False):
    """
    Perform Gradient Descent to learn theta.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix (with bias column)
    y : numpy.ndarray
        Target values
    theta : numpy.ndarray, optional
        Initial parameters (if None, initialized to zeros)
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Maximum number of iterations
    tolerance : float
        Stop if cost improvement is less than tolerance
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    theta : numpy.ndarray
        Learned parameters
    cost_history : list
        Cost at each iteration
    """
    if theta is None:
        n_features = X.shape[1]
        theta = np.zeros((n_features, 1))
    
    cost_history = []
    
    for iteration in range(n_iterations):
        # Compute gradient
        gradients = compute_gradient(X, y, theta)
        
        # Update parameters
        theta = theta - learning_rate * gradients
        
        # Compute and store cost
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        
        # Check for convergence
        if len(cost_history) > 1:
            cost_diff = abs(cost_history[-2] - cost_history[-1])
            if cost_diff < tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        # Print progress
        if verbose and iteration % 100 == 0:
            print(f'Iteration {iteration:4d} | Cost: {cost:.6f}')
    
    return theta, cost_history


def predict_proba(X, theta):
    """
    Predict probabilities using learned parameters.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix (should include bias column)
    theta : numpy.ndarray
        Learned parameters
    
    Returns:
    --------
    probabilities : numpy.ndarray
        Predicted probabilities
    """
    return sigmoid(X.dot(theta))


def predict(X, theta, threshold=0.5):
    """
    Make binary predictions using learned parameters.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix (should include bias column)
    theta : numpy.ndarray
        Learned parameters
    threshold : float
        Decision threshold
    
    Returns:
    --------
    predictions : numpy.ndarray
        Binary predictions (0 or 1)
    """
    probabilities = predict_proba(X, theta)
    return (probabilities >= threshold).astype(int)


def add_bias(X):
    """
    Add bias term (column of ones) to feature matrix.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix of shape (m, n)
    
    Returns:
    --------
    X_b : numpy.ndarray
        Feature matrix with bias column, shape (m, n+1)
    """
    m = X.shape[0]
    return np.c_[np.ones((m, 1)), X]


def accuracy_score(y_true, y_pred):
    """
    Calculate classification accuracy.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    
    Returns:
    --------
    accuracy : float
        Accuracy score (fraction of correct predictions)
    """
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    
    Returns:
    --------
    cm : numpy.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])


def precision_recall_f1(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray
        Predicted labels
    
    Returns:
    --------
    precision : float
    recall : float
    f1 : float
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
