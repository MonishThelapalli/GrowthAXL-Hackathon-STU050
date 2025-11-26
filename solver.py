import pandas as pd
import hashlib
import numpy as np
import shap
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

USER_ID = "STU050" 

def get_sha256(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()

def clean_title_flag1(title):
    return "".join(str(title).split())[:8]

def main():
    print("Processing user: " + USER_ID)

    full_hash = get_sha256(USER_ID)
    user_hash_short = full_hash[:8].upper()
    numeric_id = "".join(filter(str.isdigit, USER_ID))
    
    print(f"Target Hash: {user_hash_short}")

    try:
        books_df = pd.read_csv(r"C:\Users\monis\growthaxl\books.csv", on_bad_lines='skip')
        reviews_df = pd.read_csv(r"C:\Users\monis\growthaxl\reviews.csv", on_bad_lines='skip')
        
        books_df.columns = books_df.columns.str.strip().str.lower()
        reviews_df.columns = reviews_df.columns.str.strip().str.lower()
        
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    text_col = None
    for col in ['text', 'review_text', 'body', 'content']:
        if col in reviews_df.columns:
            text_col = col
            break
    
    if not text_col:
        return

    match_mask = reviews_df[text_col].str.contains(user_hash_short, case=False, na=False)
    target_review_row = reviews_df[match_mask]

    if target_review_row.empty:
        print("Hash not found.")
        return

    target_review = target_review_row.iloc[0]
    
    review_title = None
    if 'title' in reviews_df.columns:
        review_title = target_review['title']
    elif 'book_title' in reviews_df.columns:
        review_title = target_review['book_title']
    
    if not review_title and 'book_id' in reviews_df.columns:
          bid = target_review['book_id']
          if 'book_id' in books_df.columns:
              bmatch = books_df[books_df['book_id'] == bid]
              if not bmatch.empty:
                  review_title = bmatch.iloc[0]['title']

    if not review_title:
        review_title = "Unknown"

    flag1 = get_sha256(clean_title_flag1(review_title))
    flag2 = f"FLAG2{{{user_hash_short}}}"

    print(f"Title: {review_title}")

    book_reviews = reviews_df[reviews_df['title'] == review_title].copy()
    
    superlatives = ['perfect', 'amazing', 'best', 'excellent', 'incredible', 'fantastic', 'wonderful', 'sublime', 'superb']

    def label_review(row):
        txt = str(row.get(text_col, ''))
        words = txt.split()
        try: r = float(row.get('rating', 5))
        except: r = 5.0
            
        is_5_star = (r == 5.0)
        is_short = (len(words) < 20)
        has_sup = any(s in txt.lower() for s in superlatives)
        
        return 1 if (is_5_star and is_short and has_sup) else 0

    book_reviews['label'] = book_reviews.apply(label_review, axis=1)

    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X_sparse = tfidf.fit_transform(book_reviews[text_col].fillna(""))
    X = X_sparse.toarray()
    y = book_reviews['label'].values

    if len(np.unique(y)) < 2:
        lengths = book_reviews[text_col].str.len()
        median_len = lengths.median()
        y = np.where(lengths > median_len, 0, 1)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    genuine_indices = np.where(y == 0)[0]
    if len(genuine_indices) == 0:
        genuine_indices = np.arange(len(y))

    X_genuine = X[genuine_indices]
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_genuine, check_additivity=False)

    if isinstance(shap_values, list):
        vals = shap_values[1]
    elif len(np.shape(shap_values)) == 3:
        vals = shap_values[:, :, 1]
    else:
        vals = shap_values

    mean_shap = np.mean(vals, axis=0)
    feature_names = tfidf.get_feature_names_out()
    top_indices = np.argsort(mean_shap)[:3]
    
    top_words = [str(feature_names[i]) for i in top_indices]
    
    concat_str = "".join(top_words) + numeric_id
    flag3_hash = get_sha256(concat_str)[:10]
    flag3 = f"FLAG3{{{flag3_hash}}}"

    output = f"FLAG1 = {flag1}\nFLAG2 = {flag2}\nFLAG3 = {flag3}\n"
    with open("flags.txt", "w") as f:
        f.write(output)
    
    print("Flags generated successfully.")

if __name__ == "__main__":
    main()
