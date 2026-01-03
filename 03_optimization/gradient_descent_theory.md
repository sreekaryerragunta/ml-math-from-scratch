# Gradient Descent - Theory

## What is Gradient Descent?

Gradient descent is an **optimization algorithm** used to minimize a cost function by iteratively moving in the direction of steepest descent.

It's the **foundation** of training in machine learning and deep learning.

---

## The Core Idea

Imagine you're on a mountain (cost function surface) and want to reach the valley (minimum cost).

**Strategy**:
1. Look around to find which direction is downhill (compute gradient)
2. Take a step in that direction (update parameters)
3. Repeat until you reach the bottom (convergence)

---

## Mathematical Formulation

**Goal**: Minimize cost function $J(\theta)$

**Update Rule**:
$$\theta := \theta - \alpha \nabla J(\theta)$$

Where:
- $\theta$ = parameters (weights)
- $\alpha$ = learning rate (step size)
- $\nabla J(\theta)$ = gradient (vector of partial derivatives)

**Gradient** points in the direction of **steepest increase**, so we move in the **opposite direction** (negative gradient).

---

## Three Variants of Gradient Descent

### 1. Batch Gradient Descent (BGD)

**Uses ALL training examples** in each iteration.

**Algorithm**:
```
for iteration in 1 to max_iterations:
    θ := θ - α * (1/m) * Σ(h(x_i) - y_i) * x_i  (sum over ALL m examples)
```

**Gradient**:
$$\nabla J(\theta) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

**Pros**:
- Stable convergence (smooth path to minimum)
- Accurate gradient calculation
- Guaranteed to converge to global minimum for convex functions

**Cons**:
- Slow for large datasets (must process ALL data per update)
- Memory intensive (must load all data)
- Cannot handle online learning (streaming data)

**When to use**: Small to medium datasets (< 10,000 examples)

---

### 2. Stochastic Gradient Descent (SGD)

**Uses ONE random training example** per iteration.

**Algorithm**:
```
for iteration in 1 to max_iterations:
    randomly shuffle training data
    for i in 1 to m:
        θ := θ - α * (h(x_i) - y_i) * x_i  (only example i)
```

**Gradient (approximation)**:
$$\nabla J(\theta) \approx (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

**Pros**:
- Very fast updates (one example at a time)
- Can escape shallow local minima (due to noise)
- Works with online/streaming data
- Low memory requirements

**Cons**:
- Noisy updates (gradient is approximate)
- Never truly converges (oscillates around minimum)
- May need learning rate decay

**When to use**: Large datasets (millions of examples), online learning

---

### 3. Mini-Batch Gradient Descent

**Uses a SMALL BATCH** of training examples per iteration.

**Algorithm**:
```
for iteration in 1 to max_iterations:
    randomly shuffle training data
    for each batch of size b:
        θ := θ - α * (1/b) * Σ(h(x_i) - y_i) * x_i  (sum over batch)
```

**Gradient**:
$$\nabla J(\theta) \approx \frac{1}{b} \sum_{i=1}^{b} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)}$$

**Common batch sizes**: 32, 64, 128, 256

**Pros**:
- Balance between speed and stability
- Vectorized operations (GPU-friendly!)
- Less noisy than SGD, faster than BGD
- Industry standard (PyTorch, TensorFlow default)

**Cons**:
- Requires choosing batch size (hyperparameter)
- Still has some noise (but less than SGD)

**When to use**: Almost always! Default choice for deep learning

---

## Comparison Table

| Aspect | Batch GD | Stochastic GD | Mini-Batch GD |
|--------|----------|---------------|---------------|
| Examples per update | All (m) | 1 | b (32-256) |
| Speed per iteration | Slow | Very fast | Fast |
| Convergence path | Smooth | Noisy | Moderately smooth |
| Memory usage | High | Low | Medium |
| GPU efficiency | Poor | Poor | Excellent |
| Final convergence | Exact minimum | Oscillates | Very close |
| Typical use case | Small data | Online learning | Deep learning |

---

## Learning Rate ($\alpha$)

The learning rate controls **how big** each step is.

### Too Small
- Convergence is very slow
- May get stuck before reaching minimum
- Wastes computational resources

### Too Large
- Overshoots the minimum
- Cost may increase
- May diverge (explode)

### Just Right
- Fast convergence
- Stable updates
- Reaches minimum efficiently

**Common values**: 0.001, 0.01, 0.1

**Advanced**: Use learning rate schedules or adaptive methods (Adam, RMSprop)

---

## Convergence Criteria

When to stop gradient descent?

### 1. Maximum Iterations
Stop after fixed number of iterations.

### 2. Cost Threshold
Stop when $J(\theta) < \epsilon$ (some small value).

### 3. Gradient Magnitude
Stop when $||\nabla J(\theta)|| < \epsilon$ (gradient nearly zero).

### 4. Cost Change
Stop when $|J^{(t)} - J^{(t-1)}| < \epsilon$ (cost not improving).

**Practical**: Combine multiple criteria (e.g., max iterations AND cost threshold).

---

## Momentum

**Problem**: Standard GD can oscillate in ravines (steep in one direction, shallow in another).

**Solution**: Add momentum term that accumulates past gradients.

**Update**:
$$v := \beta v + \alpha \nabla J(\theta)$$
$$\theta := \theta - v$$

Where:
- $v$ = velocity (exponential moving average of gradients)
- $\beta$ = momentum coefficient (typically 0.9)

**Effect**: Smooths updates, accelerates convergence in consistent directions.

---

## Advanced Optimizers

Modern deep learning uses adaptive learning rates:

### Adam (Adaptive Moment Estimation)
- Combines momentum + adaptive learning rates
- Most popular in deep learning
- Works well with minimal tuning

### RMSprop
- Adapts learning rate per parameter
- Good for recurrent neural networks

### AdaGrad
- Larger updates for infrequent features
- Good for sparse data

---

## Challenges and Solutions

### Problem 1: Saddle Points
**Issue**: Gradient is zero, but not a minimum  
**Solution**: Momentum helps escape saddle points

### Problem 2: Local Minima
**Issue**: Get stuck in suboptimal minimum  
**Solution**: Use SGD (noise helps escape), or initialize differently

### Problem 3: Vanishing/Exploding Gradients
**Issue**: Gradients become too small or too large  
**Solution**: Gradient clipping, better initialization, batch normalization

### Problem 4: Slow Convergence
**Issue**: Takes too many iterations  
**Solution**: Better learning rate, use momentum, adaptive optimizers

---

## Key Insights

1. **Batch GD**: Slow but stable, good for theory
2. **SGD**: Fast but noisy, good for large-scale
3. **Mini-Batch**: Best of both worlds, industry standard
4. **Learning rate** is the most important hyperparameter
5. **Momentum** dramatically improves convergence
6. Modern optimizers (Adam) combine the best ideas

---

## Interview Questions

**Q: What's the difference between batch and stochastic gradient descent?**
A: Batch GD uses all examples per update (slow, stable). SGD uses one example (fast, noisy). Mini-batch uses small batches (balanced).

**Q: Why use mini-batch instead of batch GD?**
A: Mini-batch is much faster, GPU-efficient, and converges well. Batch GD is too slow for large datasets.

**Q: What if gradient descent doesn't converge?**
A: Check: (1) Learning rate too large? (2) Feature scaling needed? (3) Bug in gradient calculation? (4) Non-convex function with bad initialization?

---

**Next**: See implementation notebooks for hands-on examples and visualizations!
