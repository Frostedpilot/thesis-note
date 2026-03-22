

## Definition

**Kullback-Leibler Divergence** is a fundamental measure in mathematical statistics and information theory that quantifies how one probability distribution ($P$) differs from a second, reference probability distribution ($Q$).

It is often interpreted as the measure of "surprise" or the expected excess code length required to transmit a message when using an optimized code based on distribution $Q$, rather than the true distribution $P$.

## Mathematical Formulation

For two probability distributions $P$ and $Q$ defined on the same probability space $\mathcal{X}$:

**Discrete Distributions:**

  

$$D_{KL}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log\left(\frac{P(x)}{Q(x)}\right)$$

**Continuous Distributions:**

  

$$D_{KL}(P \parallel Q) = \int_{-\infty}^{\infty} p(x) \log\left(\frac{p(x)}{q(x)}\right) \, dx$$

_(where_ $p$ _and_ $q$ _are the probability density functions of_ $P$ _and_ $Q$_)_

## Key Properties

- **Non-negativity (Gibbs' Inequality):** $D_{KL}(P \parallel Q) \geq 0$  
    
    - $D_{KL}(P \parallel Q) = 0$ if and only if $P = Q$ almost everywhere.
        
- **Asymmetry:** $D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$  
    
    - Because it is not symmetric and does not satisfy the triangle inequality, KL Divergence is **not** a true statistical distance/metric.
        
- **Base of Logarithm:** The choice of logarithm base determines the unit. Base 2 yields bits (commonly used in information theory), while base $e$ yields nats.
    

## Relation to Entropy and Cross-Entropy

KL Divergence is intimately related to Shannon Entropy ($H(P)$) and Cross-Entropy ($H(P, Q)$):

  

$$D_{KL}(P \parallel Q) = H(P, Q) - H(P)$$

In machine learning, minimizing the cross-entropy loss is mathematically equivalent to minimizing the KL Divergence from the true distribution to the predicted distribution.

## Common Applications

- **Machine Learning (Loss Functions):** Used as a loss function in classification tasks and generative models like [[Variational Autoencoders]] (VAEs).
    
- **Dimensionality Reduction:** Heavily utilized in [[t-SNE]] to measure the divergence between the high-dimensional and low-dimensional probability distributions of data points.
    
- **Bayesian Inference:** Measures the information gained when revising one's beliefs from the prior distribution $Q$ to the posterior distribution $P$.
    

**See Also:**

- [[Bhattacharyya Coefficient]]
    
- [[Jensen-Shannon Divergence]] _(a symmetric, smoothed version of KL Divergence)_