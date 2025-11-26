# GrowthAXL Hackathon – STU050
**Capture The Flag (CTF) — AI Review Manipulation Detection**

This repository contains the solution for the GrowthAXL Hackathon — STU050 Capture The Flag challenge.  
The objective is to identify a manipulated book, extract hidden clues from a fake review, and compute three flags (FLAG1, FLAG2, FLAG3) using NLP, TF-IDF modeling, and SHAP interpretability.

---

##  Overview of the Solver

The solver performs three key tasks:

### 🔹 Step 1: Identify Manipulated Book (FLAG1)
- Compute SHA256 of the user ID (`STU050`)  
- Extract first 8 characters → target hash  
- Search all reviews for this embedded hash  
- Identify the corresponding book title  
- Take the first 8 non-space letters of the title  
- Compute SHA256 → `FLAG1`  

### 🔹 Step 2: Extract the Fake Review Hash (FLAG2)
- Once the review containing the hash is found:  
  `FLAG2 = FLAG2{YOUR_8_CHAR_HASH}`  

### 🔹 Step 3: SHAP-Based Authenticity Analysis (FLAG3)
- Collect all reviews of the target book  
- Apply labels:  
  - **Suspicious** = 5-star + short + superlatives  
  - **Genuine** = all others  
- Convert text to TF-IDF vectors  
- Train a RandomForest classifier  
- Run SHAP on genuine reviews only  
- Pick top 3 words that reduce suspicion (most negative SHAP values)  
- Concatenate them + numeric ID → hash → first 10 chars  
- Format: `FLAG3{xxxxxxxxxx}`  

---

##  Tech Stack

- Python 3.x  
- pandas  
- numpy  
- nltk  
- scikit-learn  
- shap  
- TF-IDF Vectorizer  

---

## 📁 Repository Structure

