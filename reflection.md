### **2. reflection.md**

```markdown
### CTF Challenge Reflection - STU050

The primary challenge in this task was extracting domain-specific keywords without a ground-truth dataset indicating which reviews were "genuine" and which were "fake." To solve this, I framed the problem as a supervised learning task using weak supervision.

I started by defining a strict heuristic for "fake" reviews based on common patterns in generated spam: short length, perfect ratings, and an over-reliance on generic superlatives like "amazing" or "perfect." This allowed me to label a subset of the data as class 1 (fake) and the rest as class 0 (genuine). While this labeling isn't perfect, it provides enough signal for a classifier to pick up on linguistic patterns.

For the model, I chose a Gradient Boosting Classifier over simpler linear models. Gradient boosting is highly effective on tabular and sparse data (like TF-IDF vectors) and can capture non-linear relationships between words better than Logistic Regression.

However, the model's accuracy was secondary to its interpretability. To extract the flag, I needed to know which words made the model classify a review as genuine. I used SHAP (SHapley Additive exPlanations) values rather than raw feature importance. SHAP provides a more granular view, allowing me to isolate words that specifically drive predictions away from the "fake" class. By filtering the top SHAP features for the "genuine" class and removing the generic superlatives, I successfully isolated the domain-specific terms required for Flag 3.
