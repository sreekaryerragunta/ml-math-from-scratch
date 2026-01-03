# Logistic Regression - Theory

## What is Logistic Regression?

Logistic regression is a **supervised learning** algorithm used for **binary classification** problems.

Despite the name, it's a **classification algorithm**, not regression!

---

## The Problem

Given training data: $(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)$ where $y \in \{0, 1\}$

We want to predict the **probability** that an example belongs to class 1:

$$P(y=1 | x; \theta)$$

Examples:
- Email: spam (1) or not spam (0)
- Tumor: malignant (1) or benign (0)  
- Student: pass (1) or fail (0)

---

## Why Not Linear Regression?

For classification, we want outputs between 0 and 1 (probabilities).

Linear regression outputs: $h_\theta(x) = \theta^T x$ can be any value $(-\infty, +\infty)$

**Problems**:
- Outputs not bounded between 0 and 1  
- Sensitive to outliers
- No probabilistic interpretation

---

## The Sigmoid (Logistic) Function

To get probabilities, we use the **sigmoid function** (also called logistic function):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties**:
1. **Range**: $\sigma(z) \in (0, 1)$ for all $z$
2. **Limits**:
   - As $z \to +\infty$: $\sigma(z) \to 1$
   - As $z \to -\infty$: $\sigma(z) \to 0$
3. **Midpoint**: $\sigma(0) = 0.5$
4. **Symmetric**: $\sigma(-z) = 1 - \sigma(z)$
5. **S-shaped curve**: Smooth transition from 0 to 1

---

## Logistic Regression Hypothesis

Combine linear model with sigmoid:

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

**Interpretation**:
- $h_\theta(x)$ = probability that $y = 1$ given $x$
- If $h_\theta(x) = 0.7$, there's a 70% chance the example is class 1

**Decision boundary**:
- Predict $y = 1$ if $h_\theta(x) \geq 0.5$ (i.e., $\theta^T x \geq 0$)
- Predict $y = 0$ if $h_\theta(x) < 0.5$ (i.e., $\theta^T x < 0$)

---

## Cost Function: Why Not MSE?

For linear regression, we used:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

**Problem**: With sigmoid, this becomes **non-convex** (many local minima)!

We need a **convex** cost function for guaranteed convergence.

---

## Log Loss (Binary Cross-Entropy)

The cost for a single example:

$$\text{Cost}(h_\theta(x), y) = \begin{cases} 
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}$$

**Intuition**:
- If $y = 1$ and $h_\theta(x) = 1$: cost = $-\log(1) = 0$ (perfect!)
- If $y = 1$ and $h_\theta(x) = 0$: cost = $-\log(0) = \infty$ (terrible!)
- If $y = 0$ and $h_\theta(x) = 0$: cost = $-\log(1) = 0$ (perfect!)
- If $y = 0$ and $h_\theta(x) = 1$: cost = $-\log(0) = \infty$ (terrible!)

**Combined formula** (clever trick):

$$\text{Cost}(h_\theta(x), y) = -y \log(h_\theta(x)) - (1-y) \log(1-h_\theta(x))$$

When $y=1$, the second term vanishes. When $y=0$, the first term vanishes.

**Full cost function**:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right]$$

This is **convex**! Gradient descent will find the global minimum.

---

## Gradient Descent for Logistic Regression

The gradient has the **same form** as linear regression!

$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)}$$

In vectorized form:

$$\nabla J(\theta) = \frac{1}{m} X^T (h_\theta(X) - y)$$

Where $h_\theta(X) = \sigma(X\theta)$ (apply sigmoid element-wise).

**Update rule**:

$$\theta := \theta - \alpha \nabla J(\theta)$$

**Note**: The formula looks identical to linear regression, but $h_\theta(x)$ is now the sigmoid function!

---

## Decision Boundary

The decision boundary is where $\theta^T x = 0$ (i.e., $h_\theta(x) = 0.5$).

**Linear decision boundary**:
- For 2 features: $\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0$ is a **line**

**Non-linear decision boundaries**:
- Add polynomial features: $x_1^2, x_2^2, x_1 x_2$, etc.
- Decision boundary can be circles, ellipses, or complex shapes

---

## Multiclass Classification

Logistic regression is for **binary** classification (2 classes).

For **multiclass** (K classes), use:
1. **One-vs-Rest (OvR)**: Train K binary classifiers
2. **Softmax Regression**: Generalization of logistic regression

---

## When Does Logistic Regression Work Well?

- **Linearly separable** classes (or use polynomial features)
- **Binary** classification
- Need **probabilistic** outputs
- Dataset is **not too large** (otherwise use SGD)
- Features are **independent** (no multicollinearity)

---

## When Does It Fail?

- **Non-linearly separable data** (use SVM with kernels or neural networks)
- Heavily **imbalanced** classes (use class weights or resampling)
- **Outliers** (less sensitive than linear regression, but still affected)
- **Perfect separation** (parameters blow up - use regularization)

---

## Regularization

To prevent overfitting, add penalty term:

**L2 (Ridge)**:
$$J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

---

## Key Takeaways

1. Logistic regression is a **classification** algorithm, not regression
2. Uses **sigmoid function** to output probabilities
3. Cost function is **log loss** (cross-entropy), not MSE
4. Gradient has same form as linear regression, but $h_\theta$ is sigmoid
5. Decision boundary is where $\theta^T x = 0$
6. Can handle non-linear boundaries with polynomial features

---

**Next**: See [derivation.ipynb](derivation.ipynb) for complete mathematical derivations.
