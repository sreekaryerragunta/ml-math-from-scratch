# PCA - Theory

## What is Principal Component Analysis (PCA)?

PCA is a **dimensionality reduction** technique that transforms high-dimensional data into a lower-dimensional space while preserving as much variance as possible.

**Core idea**: Find new axes (principal components) that capture maximum variance in the data.

---

## Why Dimensionality Reduction?

### Problems with High-Dimensional Data:
1. **Curse of dimensionality**: Data becomes sparse
2. **Visualization**: Can't plot more than 3D
3. **Computation**: Slow training with many features
4. **Overfitting**: Too many features relative to examples
5. **Storage**: Large memory requirements

### Benefits of PCA:
- Compress data (fewer features)
- Visualize high-D data in 2D/3D
- Remove noise
- Speed up learning algorithms
- Discover hidden patterns

---

## Mathematical Foundation

### Variance and Covariance

**Variance** measures spread of single variable:
$$\text{Var}(X) = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)} - \bar{x})^2$$

**Covariance** measures relationship between two variables:
$$\text{Cov}(X, Y) = \frac{1}{m}\sum_{i=1}^{m}(x^{(i)} - \bar{x})(y^{(i)} - \bar{y})$$

**Covariance Matrix** for n features:
$$\Sigma = \frac{1}{m}X^T X$$
(assuming data is centered: mean = 0)

---

## Eigenvalues and Eigenvectors

For a square matrix $A$, an eigenvector $v$ and eigenvalue $\lambda$ satisfy:
$$Av = \lambda v$$

**Interpretation**:
- Eigenvector: Direction that is only scaled (not rotated) by transformation
- Eigenvalue: How much scaling occurs

**For covariance matrix**:
- Eigenvectors: Directions of maximum variance (principal components)
- Eigenvalues: Amount of variance in each direction

---

## PCA Algorithm

### Step 1: Standardize the Data
$$x' = \frac{x - \mu}{\sigma}$$

Why? Features on different scales would dominate.

### Step 2: Compute Covariance Matrix
$$\Sigma = \frac{1}{m}X^TX$$

### Step 3: Compute Eigenvalues and Eigenvectors
Solve: $\Sigma v = \lambda v$

### Step 4: Sort by Eigenvalues
Order eigenvectors by eigenvalue (descending).

### Step 5: Select Top k Components
Choose first k eigenvectors (where most variance is).

### Step 6: Transform Data
$$X_{\text{reduced}} = X \cdot W_k$$

Where $W_k$ is matrix of top k eigenvectors.

---

## Choosing Number of Components (k)

### Method 1: Variance Explained
Keep components until cumulative variance ≥ 95% (or desired threshold).

$$\text{Variance Explained} = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{n}\lambda_i}$$

### Method 2: Scree Plot
Plot eigenvalues, look for "elbow" where curve flattens.

### Method 3: Application-Specific
- Visualization: k = 2 or 3
- Compression: Choose based on storage constraints
- Preprocessing: Based on downstream task performance

---

## PCA Properties

### What PCA Does:
- Finds orthogonal (perpendicular) directions
- First PC captures most variance
- Each subsequent PC captures most remaining variance
- PCs are uncorrelated

### What PCA Doesn't Do:
- **Not** a classification algorithm
- Doesn't use labels (unsupervised)
- Doesn't guarantee better performance (may lose important info)
- Linear transformation only (non-linear patterns need kernel PCA)

---

## Applications

### 1. Eigenfaces (Face Recognition)
- Face images: 64×64 = 4096 dimensions
- PCA reduces to ~50 dimensions
- Captures main facial features
- Fast comparison and recognition

### 2. Data Compression
- Store fewer numbers
- Lossy but controlled (know variance retained)

### 3. Visualization
- Project high-D data to 2D/3D for plotting  
- Discover clusters and patterns

### 4. Noise Reduction
- Keep components with high variance (signal)
- Discard components with low variance (noise)

### 5. Preprocessing for ML
- Speed up algorithms (fewer features)
- Reduce overfitting
- Remove multicollinearity

---

## PCA vs Other Techniques

| Method | Type | Linear | Uses Labels | Best For |
|--------|------|--------|-------------|----------|
| PCA | Unsupervised | Yes | No | Compression, visualization |
| LDA | Supervised | Yes | Yes | Classification |
| t-SNE | Unsupervised | No | No | Visualization only |
| Autoencoders | Unsupervised | No | No | Complex non-linear patterns |

---

## Limitations

### 1. Linearity
PCA assumes linear combinations. For non-linear patterns, use:
- Kernel PCA
- Autoencoders
- t-SNE (visualization only)

### 2. Interpretability
New features are combinations of originals - hard to interpret.

### 3. Sensitive to Scaling
MUST standardize features first!

### 4. Variance ≠ Information
High variance direction may not be most useful for your task.

### 5. Computational Cost
Eigendecomposition is $O(n^3)$ for n features.

---

## Reconstruction Error

After reducing dimensions, can reconstruct approximate original:
$$X_{\text{reconstructed}} = X_{\text{reduced}} \cdot W_k^T$$

**Reconstruction Error**:
$$\text{Error} = ||X - X_{\text{reconstructed}}||^2$$

Measures information loss from dimensionality reduction.

---

## When to Use PCA

### Use When:
- Too many features (> 50)
- Features are correlated
- Need visualization
- Want faster training
- Compression needed

### Don't Use When:
- Already have few features (< 10)
- Features are already uncorrelated
- Interpretability is critical
- Non-linear patterns (use kernel PCA instead)
- For supervised learning where labels should guide reduction (use LDA instead)

---

## PCA in Practice

### 1. Data Preparation
- Standardize features (CRITICAL!)
- Handle missing values first
- Check for outliers

### 2. Determine k
- Start with scree plot
- Try 95% variance threshold
- Or use cross-validation for downstream task

### 3. Transform
- Fit on training data only
- Transform both train and test with same PCs

### 4. Inverse Transform (if needed)
- Can reconstruct original space
- Useful for visualization of what PCA captured

---

## Mathematical Details

### SVD Connection
PCA via Singular Value Decomposition (more numerically stable):
$$X = U\Sigma V^T$$

Principal components are columns of V.

### Whitening
After PCA, can "whiten" data (make variance = 1 in all directions):
$$X_{\text{white}} = X_{\text{PCA}} \cdot \text{diag}(1/\sqrt{\lambda_i})$$

Useful for some ML algorithms.

---

## Key Insights

1. **PCA finds directions of maximum variance**
2. **Eigenvectors of covariance matrix = principal components**  
3. **Eigenvalues = variance in each component**
4. **Data must be standardized first**
5. **Unsupervised** - doesn't use labels
6. **Linear technique** - for non-linear, use kernel PCA
7. **Trade variance explained vs dimensionality**

---

## Interview Questions

**Q: Explain PCA in simple terms.**
A: PCA finds new axes that capture maximum variance in the data. The first axis points in the direction of highest variance, the second in the next highest (perpendicular to first), and so on. We keep only the top k axes to reduce dimensions while retaining most information.

**Q: When would you use PCA?**
A: When I have high-dimensional data (many features), especially if correlated, and want to (1) visualize in 2D/3D, (2) speed up learning, (3) reduce storage, or (4) combat overfitting.

**Q: What's the difference between PCA and LDA?**
A: PCA is unsupervised (ignores labels) and maximizes variance. LDA is supervised (uses labels) and maximizes class separation. Use PCA for compression/visualization, LDA for classification.

---

**Next**: See implementation notebooks for eigenfaces and hands-on examples!
