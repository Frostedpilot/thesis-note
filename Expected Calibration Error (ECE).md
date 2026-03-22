

## Definition

**Expected Calibration Error (ECE)** is a prominent metric used to measure how well a machine learning model's predicted probabilities align with the true empirical frequencies of the outcomes. It quantifies the discrepancy between model **confidence** (predicted probability) and model **accuracy** (actual correctness).

A model is perfectly calibrated if, out of all predictions where it predicts a class with 80% confidence, it is correct exactly 80% of the time. Modern deep neural networks are often highly accurate but poorly calibrated (typically overconfident), making ECE a crucial evaluation metric.

## Mathematical Formulation

To compute ECE, the predicted probabilities of a dataset of $n$ samples are partitioned into $M$ equally spaced bins (e.g., [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0] ).

For each bin $B_m$, the average confidence $\text{conf}(B_m)$ and the actual accuracy $\text{acc}(B_m)$ are calculated. The ECE is the weighted average of the absolute differences between accuracy and confidence across all bins:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:

- $n$ is the total number of samples.
    
- $|B_m|$ is the number of samples in bin $m$.
    
- $\text{acc}(B_m)$ is the proportion of correctly predicted samples in bin $m$.
    
- $\text{conf}(B_m)$ is the average predicted probability (confidence) for samples in bin $m$.
    

## Key Properties

- **Range:** $0 \leq \text{ECE} \leq 1$ (often expressed as a percentage).
    
    - $\text{ECE} = 0$: Perfect calibration.
        
    - **Higher ECE**: Indicates poorer calibration (either overconfidence or underconfidence).
        
- **Dependence on Binning:** The value of ECE is sensitive to the number of bins ($M$) and the binning strategy (e.g., equal-width vs. equal-mass binning). This is considered one of its primary mathematical limitations.
    

## Common Applications and Context

- **Safety-Critical Systems:** Essential in domains where knowing a model's uncertainty is as important as its accuracy, such as autonomous driving and medical diagnosis.
    
- **Reliability Diagrams:** ECE is essentially a scalar summary of a [[Reliability Diagram]] (calibration curve), which plots expected sample accuracy against model confidence.
    
- **Post-hoc Calibration:** If a model has a high ECE, techniques like [[Temperature Scaling]], [[Platt Scaling]], or [[Isotonic Regression]] are often applied to the model's logits to recalibrate the probabilities without changing the overall accuracy.
