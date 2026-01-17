# Prototype-based Demonstration Aspect-Based Sentiment Analysis (PD-ABSA)

## 📌 Overview
Aspect-Based Sentiment Analysis (ABSA) focuses on identifying the sentiment polarity (positive, negative, or neutral) expressed toward a specific aspect within a sentence. This task becomes challenging when sentiment is expressed implicitly without explicit opinion words. This project implements the **Prototype-based Demonstration Aspect-Based Sentiment Analysis (PD-ABSA)** framework to improve implicit sentiment understanding by learning stable sentiment representations and reusing them as guidance during prediction.

The implementation is developed in **Google Colab** using **PyTorch** and **Hugging Face Transformers**, and it is evaluated on standard ABSA benchmark datasets.

---

## 🧠 Key Idea
Instead of relying only on attention mechanisms or syntactic dependency structures, the PD-ABSA approach:
- Learns **sentiment prototypes** for each polarity (positive, negative, neutral)
- Uses these prototypes as **neural demonstrations** during inference
- Reduces neutral bias and improves class separability, especially for implicit sentiment

---

## 🗂️ Dataset
This project uses the **SemEval-2014 Task 4 Aspect-Based Sentiment Analysis** datasets.

- Domains: **Laptop** and **Restaurant**
- Sentiment classes used: **Positive, Negative, Neutral**
- Conflict-labeled samples are removed to maintain a clean 3-class setup

📎 Dataset Link:  
https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis

---

## ⚙️ Tools and Technologies
- **Execution Environment:** Google Colab  
- **Hardware Accelerator:** NVIDIA Tesla T4 GPU  
- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Pretrained Models:** T5, BERT (Hugging Face Transformers)  
- **Data Processing:** Pandas, NumPy  
- **Evaluation Metrics:** Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Dataset Storage:** Google Drive  

---

## 🏗️ Methodology
The PD-ABSA model follows a **two-stage learning process**:

### 1️⃣ Prototype Learning
- Input sentences are reformulated using a prompt:  
  `emotion of <aspect> : [MASK]`
- A pretrained **T5 encoder** extracts aspect-specific sentiment representations
- **Mask-aware attention** aggregates relevant contextual information
- **Contrastive learning** groups similar sentiments and separates different ones
- Stable sentiment prototypes for each class are learned

### 2️⃣ Prototype-based Demonstration
- During inference, the most relevant prototype is selected using similarity matching
- The selected prototype is combined with the input representation
- Sentiment is predicted using prototype guidance and sentiment label words

---

## 📊 Evaluation
The proposed PD-ABSA model is evaluated using:
- **Accuracy**
- **Classification Report (Precision, Recall, F1-score)**
- **Confusion Matrix**
- **ROC–AUC Curves**

### 🔍 Baseline Comparison
The performance of PD-ABSA is compared with a **DualGCN + BERT\*** baseline to highlight improvements in:
- Implicit sentiment handling
- Class separability
- Reduction of neutral bias

---

## 📈 Results Summary
- **PD-ABSA Validation Accuracy:** 96.33%
- **ROC–AUC (Macro OvR):**
  - PD-ABSA: 0.9889
  - DualGCN + BERT*: 0.5356
- Confusion matrix analysis shows strong diagonal dominance for PD-ABSA with minimal misclassification

---

## 🎯 Learning Outcomes
- Demonstrated the importance of **prototype learning** for implicit sentiment analysis
- Showed how **neural demonstrations** improve robustness and reliability
- Gained practical experience with pretrained language models and evaluation techniques for ABSA

---

## 📁 Repository Structure

