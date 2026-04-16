"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
import re
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize 
from flask import send_from_directory, request, jsonify

from models import db, AitaPost

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

_index = None  # loaded once from disk
_tfidf_cache = None


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

def _post_text(post):
    return f"{post.title} {post.selftext}"

def _build_tfidf_l2_rows(tokenized_docs):
    n_docs = len(tokenized_docs)
    vocab = sorted({t for doc in tokenized_docs for t in doc})
    V = len(vocab)
    if V == 0:
        z = np.zeros((n_docs, 0), dtype=np.float64)
        return {}, np.array([], dtype=np.float64), z, z

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
    """Truncated SVD on TF–IDF weights; safe rank-1 factors if the matrix is degenerate."""
    n_rows, n_cols = X_raw.shape
    if n_cols == 0:
        return (
            np.zeros((n_rows, 0), dtype=np.float64),
            np.zeros((0, 0), dtype=np.float64),
        )
    max_k = min(n_rows, n_cols) - 1
    if max_k < 1:
        # min(n,m)==1: full np.eye(V) would be huge when one doc has many terms — rank-1 factors instead.
        if n_rows == 1:
            v = X_raw[0].astype(np.float64, copy=False)
            nv = np.linalg.norm(v)
            words = (v / nv).reshape(-1, 1) if nv > 0 else np.zeros((n_cols, 1))
            docs = np.array([[1.0]], dtype=np.float64)
            return normalize(docs), normalize(words)
        if n_cols == 1:
            v = X_raw[:, 0].astype(np.float64, copy=False)
            nv = np.linalg.norm(v)
            docs = (v / nv).reshape(-1, 1) if nv > 0 else np.zeros((n_rows, 1))
            words = np.array([[1.0]], dtype=np.float64)
            return normalize(docs), normalize(words)
        raise RuntimeError("SVD fallback: unexpected matrix shape")

    k = min(k, max_k)
    docs_compressed = None
    words_compressed = None
    k_try = k
    while k_try >= 1:
        try:
            docs_compressed, _s, words_compressed = svds(
                csr_matrix(X_raw), k=k_try
            )
            break
        except Exception:
            k_try -= 1
    if docs_compressed is None:
        U, _s, Vt = np.linalg.svd(X_raw, full_matrices=False)
        kk = min(k, U.shape[1])
        docs_compressed = U[:, :kk]
        words_compressed = Vt[:kk, :].T
        return normalize(docs_compressed), normalize(words_compressed)

    # svds returns singular values in ascending order; flip to match largest-first latent dims
    docs_compressed = docs_compressed[:, ::-1]
    words_compressed = words_compressed[::-1, :].T
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

def _tfidf_index():
    global _tfidf_cache
    if _tfidf_cache is not None:
        return _tfidf_cache

    token_to_idx, idf, X, posts_meta = _load_index()

    X_dense = X.toarray()
    norms = np.linalg.norm(X_dense, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_normed = X_dense / norms

    docs_svd_normed, words_svd_normed = _build_svd(X_dense)
    _tfidf_cache = (token_to_idx, idf, X_normed, docs_svd_normed, words_svd_normed, posts_meta)
    return _tfidf_cache


def json_search(query, method='svd'):
    if not query or not query.strip():
        return []

    token_to_idx, idf, X_normed, docs_svd_normed, words_svd_normed, posts = _tfidf_index()
    if not posts or token_to_idx is None:
        return []
    if not token_to_idx or np.asarray(idf).size == 0:
        return []

    tokens = _tokenize(query)
    if method == 'tfidf':
        q = _query_tfidf_l2(tokens, token_to_idx, idf)
        sims = X_normed @ q
    else:
        q = _query_svd(tokens, token_to_idx, idf, words_svd_normed)
        sims = docs_svd_normed @ q

    order = np.argsort(sims)[::-1]
    matches = []
    for idx in order[:20]:
        post = posts[int(idx)]
        if isinstance(post, dict):
            matches.append({
                "id": post.get("id"),
                "submission_id": post.get("submission_id"),
                "title": post.get("title"),
                "selftext": post.get("selftext"),
                "score": post.get("score"),
                "similarity": float(sims[int(idx)]),
            })
        else:
            matches.append({
                "id": post.id,
                "submission_id": post.submission_id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
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
