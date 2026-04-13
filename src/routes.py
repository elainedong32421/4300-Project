"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
import re
import pickle

import numpy as np
<<<<<<< HEAD
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize 
=======
import scipy.sparse as sp
>>>>>>> 79df6ba68fa9640006b2a22a0be42149edfda8a7
from flask import send_from_directory, request, jsonify

from models import db, AitaPost

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

_index = None  # loaded once from disk

def _load_index():
    global _index
    if _index is not None:
        return _index
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.join(project_root, 'data', 'index', 'tfidf_meta.pkl')
    npz_path  = os.path.join(project_root, 'data', 'index', 'tfidf_matrix.npz')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    X = sp.load_npz(npz_path)
    _index = (meta['token_to_idx'], meta['idf'], X, meta['posts'])
    return _index


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


<<<<<<< HEAD
def _build_tfidf_l2_rows(tokenized_docs):
    n_docs = len(tokenized_docs)
    vocab = sorted({t for doc in tokenized_docs for t in doc})
    V = len(vocab)
    if V == 0:
        return {}, np.array([], dtype=np.float64), np.zeros((n_docs, 0), dtype=np.float64)

    token_to_idx = {t: i for i, t in enumerate(vocab)}
    C = np.zeros((n_docs, V), dtype=np.float64)
    for i, doc in enumerate(tokenized_docs):
        for t in doc:
            C[i, token_to_idx[t]] += 1.0

    df = np.sum(C > 0, axis=0)
    idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    tf = np.zeros_like(C)
    mask = C > 0
    tf[mask] = 1.0 + np.log(C[mask])
    X_raw = tf * idf[np.newaxis, :]

    norms = np.linalg.norm(X_raw, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    X_normed = X_raw / norms
    return token_to_idx, idf, X_normed, X_raw


def _query_tfidf_l2(tokenized_q, token_to_idx, idf):
    V = idf.shape[0]
    q = np.zeros(V, dtype=np.float64)
    for t in tokenized_q:
=======
def _query_vec(tokens, token_to_idx, idf):
    V = len(idf)
    q = np.zeros(V, dtype=np.float32)
    for t in tokens:
>>>>>>> 79df6ba68fa9640006b2a22a0be42149edfda8a7
        j = token_to_idx.get(t)
        if j is not None:
            q[j] += 1.0
    mask = q > 0
    q[mask] = (1.0 + np.log(q[mask])) * idf[mask]
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q

def _build_svd(X_raw, k=40):
    k = min(k, min(X_raw.shape) - 1)
    docs_compressed, s, words_compressed = svds(csr_matrix(X_raw), k=k)
    docs_compressed = docs_compressed[:, ::-1]
    words_compressed = words_compressed[::-1, :].T  # (V, k)
    return normalize(docs_compressed), normalize(words_compressed)


def _query_svd(tokenized_q, token_to_idx, idf, words_normed):
    V = idf.shape[0]
    q = np.zeros(V, dtype=np.float64)
    for t in tokenized_q:
        j = token_to_idx.get(t)
        if j is not None:
            q[j] += 1.0
    mask = q > 0
    tf_q = np.zeros_like(q)
    tf_q[mask] = 1.0 + np.log(q[mask])
    q = tf_q * idf
    q_svd = q @ words_normed  # project into latent space
    n = np.linalg.norm(q_svd)
    return q_svd / n if n > 0 else q_svd

<<<<<<< HEAD
def _tfidf_index():
    global _tfidf_cache
    n = AitaPost.query.count()
    if _tfidf_cache is not None and _tfidf_cache[0] == n:
        return _tfidf_cache[1:]

    posts = AitaPost.query.all()
    if not posts:
        _tfidf_cache = (0, None, None, None, None, None, [])
        return None, None, None, None, None, []

    tokenized = [_tokenize(_post_text(post)) for post in posts]
    token_to_idx, idf, X_normed, X_raw = _build_tfidf_l2_rows(tokenized)
    docs_svd_normed, words_svd_normed = _build_svd(X_raw)
    _tfidf_cache = (n, token_to_idx, idf, X_normed, docs_svd_normed, words_svd_normed, posts)
    return token_to_idx, idf, X_normed, docs_svd_normed, words_svd_normed, posts


def json_search(query, method='svd'):
    if not query or not query.strip():
        return []

    token_to_idx, idf, X_normed, docs_svd_normed, words_svd_normed, posts = _tfidf_index()
    if not posts or token_to_idx is None:
        return []

    tokens = _tokenize(query)
    if method == 'tfidf':
        q = _query_tfidf_l2(tokens, token_to_idx, idf)
        sims = X_normed @ q
    else:
        q = _query_svd(tokens, token_to_idx, idf, words_svd_normed)
        sims = docs_svd_normed @ q
=======
def json_search(query):
    if not query or not query.strip():
        return []

    token_to_idx, idf, X, posts = _load_index()
    q = _query_vec(_tokenize(query), token_to_idx, idf)
    sims = X.dot(q)
    order = np.argsort(sims)[::-1]
>>>>>>> 79df6ba68fa9640006b2a22a0be42149edfda8a7

    order = np.argsort(sims)[::-1]
    matches = []
    for idx in order[:20]:
        post = posts[int(idx)]
        matches.append({
<<<<<<< HEAD
            "id": post.id,
            "submission_id": post.submission_id,
            "title": post.title,
            "selftext": post.selftext,
            "score": post.score,
=======
            "id": post['id'],
            "submission_id": post['submission_id'],
            "title": post['title'],
            "selftext": post['selftext'],
            "score": post['score'],
>>>>>>> 79df6ba68fa9640006b2a22a0be42149edfda8a7
            "similarity": float(sims[int(idx)]),
        })
    return matches


def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/search")
    def search():
        query = request.args.get("query", "")
        method = request.args.get("method", "svd")
        return jsonify(json_search(query, method))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
