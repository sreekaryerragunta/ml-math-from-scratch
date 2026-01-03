# Linear Regression - Theory

## What is Linear Regression?

Linear regression is a **supervised learning** algorithm used to model the relationship between:
- **Input** (independent variable): $x$
- **Output** (dependent variable): $y$

The goal is to find a **straight line** (or hyperplane in higher dimensions) that best fits the data.

---

## The Problem

Given training data: $(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)$

We want to find a function $h(x)$ (hypothesis) that predicts $y$ from $x$.

For linear regression, the hypothesis is:

$$h_\theta(x) = \theta_0 + \theta_1 x$$

Where:
- $\theta_0$ = **intercept** (bias term)
- $\theta_1$ = **slope** (weight)
- Together, $\theta = [\theta_0, \theta_1]$ are our **parameters**

In vectorized form with multiple features:

$$h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n$$

---

## Why Do We Need It?

**Real-world applications:**
- Predicting house prices from size, location, bedrooms
- Forecasting sales based on advertising spend
- Estimating energy consumption from temperature
- Determining salary from years of experience

**Why it's fundamental:**
- Simple and interpretable
- Fast to train
- Works well when relationship is approximately linear
- Foundation for understanding more complex algorithms

---

## The Cost Function

How do we measure if our line is "good"?

We use **Mean Squared Error (MSE)**, also called the **cost function** $J(\theta)$:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

Where:
- $m$ = number of training examples
- $h_\theta(x^{(i)})$ = our prediction for example $i$
- $y^{(i)}$ = actual value for example $i$
- $(h_\theta(x^{(i)}) - y^{(i)})$ = **prediction error** (residual)

**Why square the error?**
- Without squaring, positive and negative errors would cancel out
- Squaring penalizes large errors more heavily
- Makes the math nice (differentiable, convex)

**Why divide by 2?**
- Mathematical convenience: when we take the derivative, the 2 cancels out

---

## Goal of Training

**Minimize the cost function:**

$$\min_\theta J(\theta)$$

Find the parameters $\theta$ that make our predictions as close to the actual values as possible.

---

## Two Approaches to Solve This

### 1. Closed-Form Solution (Normal Equation)

For linear regression, there's an **exact** mathematical solution:

$$\theta = (X^T X)^{-1} X^T y$$

Where:
- $X$ = design matrix (all training examples)
- $y$ = output vector

**Pros:**
- Exact answer in one step
- No hyperparameters (learning rate, iterations)

**Cons:**
- Computationally expensive for large datasets $(O(n^3))$ due to matrix inversion
- Doesn't work when $X^T X$ is not invertible
- Doesn't generalize to other algorithms

### 2. Iterative Solution (Gradient Descent)

Start with random $\theta$, then repeatedly update:

$$\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

Where:
- $\alpha$ = **learning rate** (step size)
- $\frac{\partial J(\theta)}{\partial \theta_j}$ = **gradient** (direction of steepest increase)

**Pros:**
- Works for any differentiable cost function
- Scales to large datasets
- Generalizes to neural networks, logistic regression, etc.

**Cons:**
- Requires choosing learning rate
- Requires multiple iterations
- Can get stuck in local minima (not a problem for linear regression since $J$ is convex)

---

## Intuition: How Gradient Descent Works

Imagine you're on a mountain (the cost function surface) and want to reach the valley (minimum cost).

1. **Look around** (compute gradient): Which direction is downhill?
2. **Take a step** downhill (update parameters)
3. **Repeat** until you reach the bottom (convergence)

The **learning rate** $\alpha$ controls step size:
- Too small → takes forever to converge
- Too large → overshoots the minimum, diverges

---

## Key Assumptions of Linear Regression

1. **Linearity**: Relationship between $x$ and $y$ is approximately linear
2. **Independence**: Training examples are independent
3. **Homoscedasticity**: Variance of errors is constant
4. **Normality**: Errors are normally distributed (for inference)
5. **No multicollinearity**: Features are not highly correlated (for multiple features)

---

## When Linear Regression Fails

- **Non-linear relationships**: Need polynomial features or different models
- **Outliers**: Heavily influence the line (use robust regression)
- **Heteroscedasticity**: Variance changes with $x$ (use weighted regression)
- **Overfitting**: Too many features (use regularization - Ridge, Lasso)

---

## What's Next?

In the accompanying notebooks, we will:

1. **derivation.ipynb**: Derive the gradient mathematically step-by-step
2. **linear_regression_from_scratch.ipynb**: Implement both approaches in NumPy
3. **experiments.ipynb**: Test on different datasets, visualize behavior

---

## Key Takeaways

- Linear regression finds the best-fit line by minimizing squared errors
- Two solutions: closed-form (exact, expensive) vs gradient descent (iterative, scalable)
- Gradient descent is the foundation for all deep learning
- Understanding this deeply makes you ready for complex algorithms

**Next**: See [derivation.ipynb](derivation.ipynb) for complete mathematical derivations.
