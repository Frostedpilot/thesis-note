

## Definition

The **Bhattacharyya coefficient** is a statistical measure of the amount of **overlap** or similarity between two probability distributions. It is widely used in pattern recognition, statistics, and machine learning to determine the relative closeness of two samples being considered.

## Mathematical Formulation

For two probability distributions $p$ and $q$ over the same domain $X$:

**Discrete Distributions:**

  

$$BC(p, q) = \sum_{x \in X} \sqrt{p(x) q(x)}$$

**Continuous Distributions:**

  

$$BC(p, q) = \int_{X} \sqrt{p(x) q(x)} \, dx$$

## 🔑 Key Properties

- **Range:** $0 \leq BC \leq 1$  
    
    - $BC = 0$: Absolute zero overlap (the distributions share no common domain where both have non-zero probability).
        
    - $BC = 1$: Perfect match (the distributions are identical).
        
- **Symmetry:** $BC(p, q) = BC(q, p)$. This makes it fundamentally different from metrics like [[Kullback-Leibler Divergence (KL Divergence)]], which are asymmetric.
    

## 🔗 Relation to Bhattacharyya Distance

The coefficient is used to calculate the **Bhattacharyya Distance** ($D_B$), which measures the _dissimilarity_ between the distributions:

  

$$D_B(p, q) = -\ln(BC(p, q))$$

_(Note: Despite the name, Bhattacharyya distance does not satisfy the triangle inequality, so it is not a true metric)._

## Common Applications

- **Feature Selection:** Determining which features best separate two classes (less overlap = better feature).
    
- **Computer Vision:** Image tracking and histogram matching (e.g., comparing color histograms of images).
    
- **Signal Processing:** Measuring the separability of classes in noisy signals.