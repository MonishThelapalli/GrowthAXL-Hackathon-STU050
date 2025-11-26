import pandas as pd
import hashlib
import numpy as np
import shap
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

USER_ID = "STU050" 

nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def get_hash(s):
    return hashlib.sha256(str(s).encode('utf-8')).hexdigest()

def clean_title(title):
    return re.sub(r'\s+', '', str(title))[:8]

def main():
    print(f"Processing for {USER_ID}...")

    uid_hash = get_hash(USER_ID)
    short_hash = uid_hash[:8].upper()
    num_id = "".join(filter(str.isdigit, USER_ID))
    
    print(f"Hash target: {short_hash}")

    try:
        books = pd.read_csv("books.csv", on_bad_lines='skip')
        reviews = pd.read_csv("reviews.csv", on_bad_lines='skip')
        books.columns = books.columns.str.strip().str.lower()
        reviews.columns = reviews.columns.str.strip().str.lower()
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    txt_col = next((c for c in ['text', 'review_text', 'body', 'content'] if c in reviews.columns), None)
    
    hits = reviews[reviews[txt_col].str.contains(short_hash, case=False, na=False)]

    if hits.empty:
        print("Hash not found in reviews.")
        return

    target = hits.iloc[0]
    print(f"Found review at index: {target.name}")

    book_title = "Unknown"
    related_ids = []

    if 'parent_asin' in target: related_ids.append(target['parent_asin'])
    if 'asin' in target: related_ids.append(target['asin'])

    b_row = None
    for bid in related_ids:
        if 'parent_asin' in books.columns:
            match = books[books['parent_asin'] == bid]
            if not match.empty:
                b_row = match.iloc[0]
                break
        if 'asin' in books.columns:
            match = books[books['asin'] == bid]
            if not match.empty:
                b_row = match.iloc[0]
                break

    if b_row is not None:
        book_title = b_row['title']
        same_books = books[books['title'] == book_title]
        
        if 'parent_asin' in same_books.columns:
            related_ids.extend(same_books['parent_asin'].tolist())
        if 'asin' in same_books.columns:
            related_ids.extend(same_books['asin'].tolist())
            
        related_ids = list(set(related_ids))
        print(f"Book: {book_title}")
    else:
        book_title = target.get('title', 'Unknown')

    f1 = get_hash(clean_title(book_title))
    f2 = f"FLAG2{{{short_hash}}}"
    
    print(f"F1: {f1}")
    print(f"F2: {f2}")

    if related_ids:
        col = 'parent_asin' if 'parent_asin' in reviews.columns else 'asin'
        df_train = reviews[reviews[col].isin(related_ids)].copy()
    else:
        df_train = hits.copy()

    if len(df_train) < 5 and 'book_title' in reviews.columns:
        df_train = reviews[reviews['book_title'] == book_title].copy()

    use_global = False
    if len(df_train) < 10:
        print("Not enough local data, switching to global sample.")
        use_global = True
        df_train = reviews.sample(n=min(2000, len(reviews)), random_state=42).copy()
    else:
        print(f"Training on {len(df_train)} reviews.")

    sups = ['perfect', 'amazing', 'best', 'excellent', 'incredible', 'fantastic', 'wonderful', 'sublime', 'superb']

    def get_label(row):
        t = str(row.get(txt_col, ''))
        w = t.split()
        try: r = float(row.get('rating', 5))
        except: r = 5.0
        
        if r == 5.0 and len(w) < 20 and any(s in t.lower() for s in sups):
            return 1
        return 0

    df_train['label'] = df_train.apply(get_label, axis=1)
    
    if df_train['label'].nunique() < 2:
        df_train.iloc[0, df_train.columns.get_loc('label')] = 1 - df_train.iloc[1, df_train.columns.get_loc('label')]

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
    X = vectorizer.fit_transform(df_train[txt_col].fillna("")).toarray()
    y = df_train['label'].values
    
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    
    if not use_global:
        X_ex = X[y == 0]
    else:
        X_ex = X[y == 0][:100]

    shap_vals = explainer.shap_values(X_ex, check_additivity=False)
    
    if isinstance(shap_vals, list): 
        vals = shap_vals[1]
    elif len(np.shape(shap_vals)) == 3: 
        vals = shap_vals[:, :, 1]
    else: 
        vals = shap_vals

    mean_shap = np.mean(vals, axis=0)
    sorted_idx = np.argsort(mean_shap)
    
    feat_names = vectorizer.get_feature_names_out()
    
    forbidden = set(sups)
    candidates = []
    
    print("Top words:")
    for i in sorted_idx[:20]:
        w = str(feat_names[i])
        print(f"{w} ({mean_shap[i]:.4f})")
        if w not in forbidden and len(w) > 3:
             candidates.append(w)

    top_3 = candidates[:3]
    print(f"Selected: {top_3}")
    
    cat = "".join(top_3) + num_id
    f3_hash = get_hash(cat)[:10]
    f3 = f"FLAG3{{{f3_hash}}}"
    
    result = f"FLAG1 = {f1}\nFLAG2 = {f2}\nFLAG3 = {f3}\n"
    with open("flags.txt", "w") as f:
        f.write(result)
        
    print("flags.txt saved.")

if __name__ == "__main__":
    main()
