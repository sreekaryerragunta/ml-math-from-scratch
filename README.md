# ML Math From Scratch

> Building core machine learning algorithms from first principles to demonstrate deep mathematical understanding.

## Purpose

Every algorithm is implemented using only NumPy, with complete mathematical derivations, visual intuitions, and comparisons to industry-standard libraries like scikit-learn.

---

## What's Inside

| Component | Theory | Derivation | Implementation | Focus |
|-----------|--------|------------|----------------|-------|
| **01. Linear Regression** | [theory.md](01_linear_regression/theory.md) | [derivation.ipynb](01_linear_regression/derivation.ipynb) | [implementation](01_linear_regression/linear_regression_from_scratch.ipynb) | Gradient descent, closed-form solution |
| **02. Logistic Regression** | [theory.md](02_logistic_regression/theory.md) | [derivation.ipynb](02_logistic_regression/derivation.ipynb) | [implementation](02_logistic_regression/logistic_regression_from_scratch.ipynb) | Sigmoid, cross-entropy, decision boundaries |
| **03. Optimization** | [theory.md](03_optimization/gradient_descent_theory.md) | - | [GD variants](03_optimization/gd_from_scratch.ipynb) | Batch GD, SGD, Mini-batch, convergence |
| **04. Regularization** | [theory.md](04_regularization/theory.md) | - | [L1/L2](04_regularization/l1_l2_from_scratch.ipynb) | Ridge, Lasso, bias-variance tradeoff |
| **05. PCA** | [theory.md](05_pca/theory.md) | [eigen decomposition](05_pca/eigen_decomposition.ipynb) | [PCA](05_pca/pca_from_scratch.ipynb) | Dimensionality reduction, eigenfaces |

---

## Repository Philosophy

Each component follows a consistent structure:

1. **theory.md** → High-level concepts and intuition
2. **derivation.ipynb** → Step-by-step mathematical derivations with LaTeX
3. **implementation.ipynb** → Scratch code + sklearn comparison + visualizations
4. **experiments.ipynb** → Testing edge cases and scenarios
5. **utils.py** → Clean, reusable helper functions

---

## Why This Approach?

### Shows Mathematical Maturity
- Full derivations of cost functions, gradients, and update rules
- Understanding of linear algebra (eigendecomposition) and calculus (partial derivatives)
- Ability to explain *why* algorithms work, not just *how* to use them

### Proves Engineering Skills
- Clean, documented, tested code
- Modular design with reusable utilities
- Proper comparison with production libraries

### Production Ready
- Can derive gradient descent on a whiteboard
- Can explain convergence criteria and failure modes
- Understands trade-offs (batch vs SGD, L1 vs L2, closed-form vs iterative)

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/sreekaryerragunta/ml-math-from-scratch.git
cd ml-math-from-scratch

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

Then navigate to any component folder and open the notebooks.

---

## Key Learnings

Through building this repository, I learned:

1. **Math → Code Translation**: Turning mathematical equations into working NumPy implementations
2. **Optimization Intuition**: Why gradient descent converges, when it fails, and how to fix it
3. **Implementation Choices**: Trade-offs between closed-form solutions vs iterative methods
4. **Debugging from First Principles**: When your gradient doesn't converge, you need to understand the underlying math
5. **Regularization Philosophy**: How L1/L2 prevent overfitting and when to use each

---

## What's Next?

This repository covers **mathematical foundations**. For practical applications, see:

- [core-machine-learning](https://github.com/sreekaryerragunta/core-machine-learning) - SVM, Decision Trees, Random Forests, Clustering
- [deep-learning-from-scratch](https://github.com/sreekaryerragunta/deep-learning-from-scratch) - Neural Networks, Backpropagation, CNNs
- [applied-ml-projects](https://github.com/sreekaryerragunta/applied-ml-projects) - Real-world problem solving

---

## Contact

**Sreekar Yerragunta**  
GitHub: [@sreekaryerragunta](https://github.com/sreekaryerragunta)

---

*Built with NumPy, visualized with Matplotlib, validated with scikit-learn.*

