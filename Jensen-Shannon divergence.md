

## Definition

The **Jensen-Shannon Divergence** is a popular method of measuring the similarity between two probability distributions. It is directly based on the [[Kullback-Leibler Divergence (KL Divergence)]] (KL divergence), but it resolves some of KL divergence's primary limitations by being symmetric and always yielding a finite, bounded value.

It is sometimes referred to as the Information Radius or total divergence to the average.

## Mathematical Formulation

For two probability distributions $P$ and $Q$, the Jensen-Shannon divergence is defined as:

$$JSD(P \parallel Q) = \frac{1}{2} D_{KL}(P \parallel M) + \frac{1}{2} D_{KL}(Q \parallel M)$$

Where:

- $M = \frac{1}{2}(P + Q)$ is a mixture distribution (the point-wise average of $P$ and $Q$).
    
- $D_{KL}$ is the Kullback-Leibler divergence.
    

## Key Properties

- **Symmetry:** $JSD(P \parallel Q) = JSD(Q \parallel P)$. This is a major advantage over KL divergence, making it much easier to use as a straightforward comparative measure.
    
- **Bounded Range:** $0 \leq JSD(P \parallel Q) \leq 1$ (provided the base-2 logarithm is used when calculating the underlying KL divergence).
    
    - $0$ means the distributions are completely identical.
        
    - $1$ means the distributions are entirely disjoint (no overlapping domain).
        
- **Metric Property:** While JSD itself is not a true mathematical metric, its square root $\sqrt{JSD(P \parallel Q)}$ (known as the Jensen-Shannon distance) _is_ a true metric and fully satisfies the triangle inequality.
    

## Common Applications

- **Machine Learning (GANs):** The original mathematical formulation of Generative Adversarial Networks (GANs) effectively minimizes the Jensen-Shannon divergence between the real data distribution and the model's generated distribution.
    
- **Bioinformatics:** Frequently used for comparing sequence alignments, genomes, and protein profiles due to its stability.
    
- **Natural Language Processing (NLP):** Comparing document topics, word probability distributions, and linguistic structures.
    

**See Also:**

- [[Kullback-Leibler Divergence (KL Divergence)]]
    
- [[Bhattacharyya Coefficient]]
