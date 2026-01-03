"""
Utility functions for Linear Regression

This module contains reusable functions for:
- Cost computation
- Gradient computation
- Gradient Descent optimization
- Normal Equation solver
- Model evaluation
"""

import numpy as np


def compute_cost(X, y, theta):
    """
    Compute the Mean Squared Error cost function.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix of shape (m, n+1) including bias column
    y : numpy.ndarray
        Target values of shape (m, 1)
    theta : numpy.ndarray
        Parameters of shape (n+1, 1)
    
    Returns:
    --------
    cost : float
        The MSE cost
    """
    m = len(y)
    predictions = X.dot(theta)
    errors = predictions - y
    cost = (1 / (2 * m)) * np.sum(errors ** 2)
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
    predictions = X.dot(theta)
    errors = predictions - y
    gradients = (1 / m) * X.T.dot(errors)
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
        Initial parameters (if None, initialized randomly)
    learning_rate : float
        Step size for gradient descent
    n_iterations : int
        Maximum number of iterations to run
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
    # Initialize theta if not provided
    if theta is None:
        n_features = X.shape[1]
        theta = np.random.randn(n_features, 1)
    
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


def normal_equation(X, y):
    """
    Compute optimal parameters using the Normal Equation.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix (with bias)
    y : numpy.ndarray
        Target values
    
    Returns:
    --------
    theta : numpy.ndarray
        Optimal parameters
    """
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


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


def predict(X, theta):
    """
    Make predictions using learned parameters.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Design matrix (should include bias column)
    theta : numpy.ndarray
        Learned parameters
    
    Returns:
    --------
    predictions : numpy.ndarray
        Predicted values
    """
    return X.dot(theta)


def r2_score(y_true, y_pred):
    """
    Calculate R² (coefficient of determination).
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns:
    --------
    r2 : float
        R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def mean_squared_error(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns:
    --------
    mse : float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True values
    y_pred : numpy.ndarray
        Predicted values
    
    Returns:
    --------
    mae : float
        Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))
