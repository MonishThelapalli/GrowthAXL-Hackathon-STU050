# Reflection

The main challenge in this task was extracting domain-specific keywords without access to any true labels indicating which reviews were genuine or fake. To address this, I reframed the problem as a weakly supervised learning task.

I designed a heuristic to approximate fake reviews:
- very short reviews,
- perfect 5-star ratings,
- use of generic praise terms such as “amazing” or “perfect”.

These characteristics commonly appear in spam or auto-generated reviews, allowing the creation of a proxy label:  
**Fake = 1**, **Genuine = 0**.

Once pseudo-labels were assigned, I trained a **Gradient Boosting Classifier**, chosen specifically because it handles sparse TF-IDF features well and captures non-linear relationships better than linear models like Logistic Regression.

However, the goal was not high accuracy — it was interpretability. I needed to understand **why** the classifier marked certain reviews as genuine to extract the hidden words required for Flag 3.

To achieve this, I used **SHAP values**, which provide local and global explanations for model predictions. SHAP allowed me to isolate only those words that strongly pulled predictions *away* from the fake class. After filtering out stopwords and generic superlatives, I obtained meaningful domain-specific terms that were concatenated and hashed to generate **Flag 3**.

This combined approach — heuristic labeling, interpretable machine learning, and SHAP-based feature extraction — produced a robust method to uncover all required flags.
