# Approach Summary

## Overview

This solution automates the process of identifying hidden flags within a dataset of books and reviews. It combines direct text search for static flags with a machine learning approach to extract context-specific keywords for the final flag.

---

## Methodology

### **1. User Identification & Hashing**

- The script begins by hashing the provided `USER_ID` using **SHA-256**.  
- The **first 8 characters** of this hash act as a unique identifier to locate an "injection point" inside the review dataset.

---

### **2. Data Ingestion & Linkage**

**Loading:**  
Both `books.csv` and `reviews.csv` are loaded, with column names normalized to avoid whitespace mismatches.

**Target Review Discovery:**  
The script scans the review text to find entries containing the user’s short hash.

**Book Association:**  
Once the target review is found, the script links it to the correct book using `parent_asin` or `asin`, allowing extraction of the corresponding **Book Title**.

---

### **3. Static Flag Generation**

- **Flag 1:**  
  Created by hashing the cleaned book title (spaces removed).

- **Flag 2:**  
  Constructed directly from the user's short hash in the specified problem format.

---

### **4. Machine Learning for Flag 3**

The third flag depends on identifying domain-specific words that differentiate genuine reviews from generic ones.

#### **Heuristic Labeling**
Since the dataset is unlabeled:
- Reviews with **5-star rating**, **<20 words**, and containing generic superlatives (e.g., *amazing*, *perfect*) are labeled **Fake (1)**.  
- Others are labeled **Genuine (0)**.

#### **Vectorization**
- Text is converted using **TF-IDF**, emphasizing distinct words and down-weighting common ones.

#### **Model Training**
- A **Gradient Boosting Classifier** is trained to separate fake vs genuine reviews based on the heuristic labels.

---

### **5. Feature Extraction via SHAP**

To extract the hidden words:

- **SHAP (SHapley Additive exPlanations)** is used to interpret the model's output.  
- Words with strong **negative impact on the "Fake" class** (i.e., strong indicators of genuine reviews) are retrieved.  
- The **top 3 meaningful words** (excluding generic superlatives and stopwords) are concatenated with the numeric part of the user ID and hashed to form **Flag 3**.

---
