# FedDCA: Federated Learning under Extreme Domain Skew

This repository contains the code and experimental setup for **FedDCA**, a federated learning framework designed to handle **extreme domain-skew heterogeneity** through **accuracy-based weighted aggregation** and **dynamic client- and class-level data adjustment**.

FedDCA improves both **performance** and **fairness** in highly non-IID settings, where clients share labels but differ significantly in feature distributions.

---

## 📄 Paper

**FedDCA: Dynamic Client and Class Adjustment for Extreme Domain-Skew Federated Learning**  
*Final Project – CS6304: Advanced Topics in Machine Learning*  
LUMS University

Authors:
- Hasan Hameed  
- Muhammad Zaeem Rizwan  
- Fayzan Ali Akhtar  

The full paper is included as:

```
FedDCA.pdf
```

---

## 🚩 Motivation

Standard federated learning methods such as **FedAvg** struggle under extreme domain skew due to **client drift**, where local models converge toward incompatible optima.

Real-world examples include:
- Satellite vs. drone imagery in agriculture  
- Sketch vs. real images in vision tasks  
- MNIST vs. SVHN in digit recognition  

FedDCA addresses these challenges by **adapting both aggregation and data allocation dynamically**, rather than treating all client updates equally.

---

## 🧠 Core Idea

FedDCA introduces two key mechanisms:

### 1. Accuracy-Based Weighted Aggregation
- Clients evaluate their local models on validation data
- The server assigns **higher aggregation weights** to clients with better accuracy
- This ensures **high-quality updates have stronger influence**

### 2. Dynamic Dataset & Class Adjustment
- Clients with lower performance receive **larger datasets** in subsequent rounds
- Poorly performing classes are **oversampled**
- Adjustments happen **iteratively per communication round**

---

## 🔬 Method Overview

**Per Communication Round:**
1. Server sends the global model
2. Each client:
   - Samples a subset of its local data
   - Trains a local model
   - Computes overall and class-wise accuracy
3. Clients send:
   - Model parameters
   - Accuracy metrics
4. Server:
   - Aggregates models using accuracy-based weighting
   - Adjusts dataset size and class ratios for next round

---

## 🧪 Experimental Setup

### Datasets
- **DomainNet** (Real, Sketch, Clipart, Painting, Infograph, Quickdraw)
- **Digits** (MNIST, SVHN, USPS)

### Model
- SimpleCNN
- 10 epochs per round
- 20 communication rounds
- Batch size: 128
- Learning rate: 0.001

### Baseline
- FedAvg

### Evaluation Metrics
- Overall accuracy
- Per-domain accuracy
- Per-class accuracy
- Standard deviation across domains (fairness)

---

## 📊 Key Results

- **15–20% average accuracy improvement** over FedAvg on DomainNet
- **Lower variance across domains**, indicating better fairness
- **Faster convergence** (≈ half the rounds of FedAvg)
- Robust performance under increased dataset size and class count

---

## 🔐 Privacy & Scalability

- No raw data is shared
- Only model parameters and scalar accuracy metrics are communicated
- Suitable for **edge and IoT devices**

---

## ⚠️ Limitations & Future Work

- Relies on accurate validation metrics
- Susceptible to noisy or malicious clients
- Future directions:
  - Robust weighting via anomaly detection
  - Mixed domain + label skew
  - Comparison with additional SOTA methods

---

## 📌 Citation

```bibtex
@article{feddca2025,
  title={FedDCA: Dynamic Client and Class Adjustment for Extreme Domain-Skew Federated Learning},
  author={Hameed, Hasan and Rizwan, Muhammad Zaeem and Akhtar, Fayzan Ali},
  year={2025},
  institution={LUMS University}
}
```

---

## 📬 Contact

**Hasan Hameed** — hasanhd555@gmail.com

---

## ⭐ Acknowledgements

This work was completed as part of **CS6304: Advanced Topics in Machine Learning** at **LUMS University**.
