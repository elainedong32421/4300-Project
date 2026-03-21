"""
Run once to pre-build the TF-IDF index and save it to disk.
Usage: python src/build_index.py
"""
import os
import re
import pickle
import csv
import numpy as np
from scipy.sparse import csr_matrix, save_npz

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_DIR = os.path.join(project_root, 'data', 'index')
INDEX_NPZ = os.path.join(INDEX_DIR, 'tfidf_matrix.npz')
INDEX_META = os.path.join(INDEX_DIR, 'tfidf_meta.pkl')


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def build():
    csv_path = os.path.join(project_root, 'data', 'AITA_clean1.csv')
    print("Reading CSV...")
    posts = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            posts.append({
                'id': int(row['id']) if row['id'] else 0,
                'submission_id': row['submission_id'],
                'title': row['title'],
                'selftext': row['selftext'],
                'score': int(row['score']) if row['score'] else 0,
            })
    print(f"  {len(posts)} posts loaded")

    print("Tokenizing...")
    tokenized = [tokenize(f"{p['title']} {p['selftext']}") for p in posts]

    print("Building vocab...")
    vocab = sorted({t for doc in tokenized for t in doc})
    V = len(vocab)
    token_to_idx = {t: i for i, t in enumerate(vocab)}
    print(f"  Vocab size: {V}")

    n_docs = len(posts)
    print("Building sparse TF-IDF matrix...")

    # Build sparse matrix using COO format
    rows_idx, cols_idx, vals = [], [], []
    for i, doc in enumerate(tokenized):
        counts = {}
        for t in doc:
            j = token_to_idx.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, cnt in counts.items():
            rows_idx.append(i)
            cols_idx.append(j)
            vals.append(1.0 + np.log(cnt))
        if i % 10000 == 0:
            print(f"  {i}/{n_docs}", end='\r')

    print()
    import scipy.sparse as sp
    C = sp.csr_matrix((vals, (rows_idx, cols_idx)), shape=(n_docs, V), dtype=np.float32)

    print("Computing IDF...")
    df = np.diff(C.tocsc().indptr).astype(np.float32)  # doc freq per term
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    print("Applying IDF and L2 normalizing...")
    # Multiply each row by idf
    C = C.multiply(idf[np.newaxis, :])

    # L2 normalize each row
    norms = np.sqrt(C.multiply(C).sum(axis=1)).A1
    norms = np.where(norms == 0, 1.0, norms)
    C = C.multiply(1.0 / norms[:, np.newaxis])

    print("Saving index...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    save_npz(INDEX_NPZ, C)
    with open(INDEX_META, 'wb') as f:
        pickle.dump({'token_to_idx': token_to_idx, 'idf': idf, 'posts': posts}, f)

    print(f"Done! Saved to {INDEX_DIR}")
    print(f"  Matrix shape: {C.shape}, nnz: {C.nnz}")


if __name__ == '__main__':
    build()
