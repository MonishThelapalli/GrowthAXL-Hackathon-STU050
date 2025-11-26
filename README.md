# Capture the Flag (CTF) Challenge - STU050

This repository contains the solution for the "AI Detective" Capture the Flag challenge. The objective was to identify a manipulated book rating in a large dataset, locate a hidden hash, and train a machine learning model to distinguish between suspicious and genuine reviews using SHAP analysis.

## 📂 Repository Structure

* **`solver.py`**: The main Python script that automates the entire process (Data loading, Hash detection, Model training, SHAP analysis).
* **`flags.txt`**: The final output containing the three required flags.
* **`reflection.md`**: A summary of the methodology and problem-solving process.
* **`books.csv` / `reviews.csv`**: Dataset files (not included in repo to save space, expected in local directory).

## 🛠️ Setup & Requirements

The solution requires Python 3.8+ and the following libraries:

```bash
pip install pandas numpy scikit-learn shap nltk
