# Regularization - Theory

## What is Regularization?

Regularization is a technique to **prevent overfitting** by adding a penalty term to the cost function that discourages complex models.

Core idea: **Simpler models generalize better**.

---

## The Overfitting Problem

**Overfitting**: Model fits training data perfectly but performs poorly on new data.

**Causes**:
- Too many features (high-dimensional data)
- Not enough training data
- Model is too complex (high-degree polynomials)

**Solution**: Regularization constrains model complexity.

---

## Types of Regularization

### 1. L2 Regularization (Ridge)

Add squared magnitude of coefficients to cost function.

**Modified Cost**:
$$J(\theta) = \text{Original Cost} + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

Where:
- $\lambda$ = regularization parameter (controls strength)
- Larger $\lambda$ → stronger regularization → simpler model
- Note: We DON'T regularize $\theta_0$ (bias term)

**Effect**:
- Shrinks all coefficients toward zero
- Prefers many small weights over few large weights
- Smooth, differentiable penalty

**Use when**: You want to keep all features but reduce their impact

---

### 2. L1 Regularization (Lasso)

Add absolute magnitude of coefficients to cost function.

**Modified Cost**:
$$J(\theta) = \text{Original Cost} + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

**Effect**:
- Shrinks some coefficients to **exactly zero**
- Performs **feature selection** automatically
- Creates sparse models (many zeros)

**Use when**: You suspect many features are irrelevant

---

## L1 vs L2 Comparison

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | $\|\theta\|$ | $\theta^2$ |
| Effect | Sparse (zeros) | Shrinkage |
| Feature selection | Yes | No |
| Solution | Non-differentiable at 0 | Smooth everywhere |
| When all features correlated | Picks one arbitrarily | Shrinks all equally |
| Computational | Harder | Easier |

---

## Bias-Variance Tradeoff

### Bias
**High bias** = model is too simple, underfits
- Misses patterns in data
- Poor performance on training AND test data
- Example: Linear model for non-linear data

### Variance  
**High variance** = model is too complex, overfits
- Captures noise as patterns
- Excellent on training, poor on test
- Example: High-degree polynomial

### The Tradeoff

**Regularization increases bias but decreases variance**

Optimal model balances both:
- Some bias (model not perfect on training data)
- Low variance (generalizes well to test data)

**Total Error = Bias² + Variance + Irreducible Error**

---

## Choosing Regularization Parameter (λ)

### λ = 0
- No regularization
- May overfit

### λ too small
- Minimal regularization
- Still overfits

### λ optimal
- Balanced bias-variance
- Best test performance

### λ too large
- Over-regularization
- Underfits (all weights near zero)

**How to choose**: Use cross-validation!

---

## Ridge Regression (L2) in Detail

**For Linear Regression**:

**Cost**:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$$

**Gradient**:
$$\nabla J(\theta) = \frac{1}{m} X^T(X\theta - y) + \frac{\lambda}{m}\theta$$

**Normal Equation**:
$$\theta = (X^TX + \lambda I)^{-1} X^T y$$

Note: Adding $\lambda I$ makes matrix invertible even when $X^TX$ is singular!

---

## Lasso Regression (L1) in Detail

**For Linear Regression**:

**Cost**:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|$$

**Challenge**: Not differentiable at $\theta_j = 0$

**Solution**: Use subgradient descent or coordinate descent

**Effect**: Forces some $\theta_j$ to exact zero (sparse solution)

---

## Elastic Net

Combines L1 and L2:

$$J(\theta) = \text{MSE} + \lambda_1 \sum |\theta_j| + \lambda_2 \sum \theta_j^2$$

**Benefits**:
- Feature selection (from L1)
- Stability (from L2)
- Works well when features are correlated

**Use when**: Want benefits of both L1 and L2

---

## Practical Guidelines

### When to Regularize?
- Number of features > number of examples
- Seeing overfitting (big gap between train/test error)
- Features are correlated
- Want to prevent numerical instability

### Which Regularization?
- **Default**: Try L2 (Ridge) first
- **Feature selection needed**: Use L1 (Lasso)
- **Many correlated features**: Use Elastic Net
- **Deep learning**: L2 + dropout

### Hyperparameter Tuning
Use k-fold cross-validation:
1. Try λ values: [0.001, 0.01, 0.1, 1, 10, 100]
2. For each λ, compute average validation error
3. Choose λ with lowest validation error
4. Retrain on all data with chosen λ

---

## Regularization in Logistic Regression

**L2 Regularized**:
$$J(\theta) = -\frac{1}{m}\sum [y\log(h) + (1-y)\log(1-h)] + \frac{\lambda}{2m}\sum\theta_j^2$$

**L1 Regularized**:
$$J(\theta) = -\frac{1}{m}\sum [y\log(h) + (1-y)\log(1-h)] + \frac{\lambda}{m}\sum|\theta_j|$$

Same principles apply!

---

## Key Insights

1. **Regularization prevents overfitting** by penalizing complexity
2. **L2 (Ridge)**: Shrinks weights, keeps all features
3. **L1 (Lasso)**: Creates sparse models, automatic feature selection
4. **λ controls tradeoff**: Larger λ = simpler model
5. **Use cross-validation** to choose optimal λ
6. **Bias-variance tradeoff**: Regularization increases bias, decreases variance

---

## Key Questions

**Q: What's the difference between L1 and L2 regularization?**
A: L2 shrinks all weights toward zero smoothly. L1 forces some weights to exactly zero, performing feature selection. L2 penalty is differentiable everywhere, L1 is not (at zero).

**Q: When would you use regularization?**
A: When overfitting (train error << test error), when features > examples, when features are highly correlated, or to prevent numerical instability.

**Q: How do you choose the regularization parameter?**
A: Use k-fold cross-validation. Try multiple λ values (0.001 to 100), pick the one with best validation performance.

---

**Next**: See implementation notebooks for hands-on examples!
