"""
Routes: React app serving and episode search API.

Set SPARK_API_KEY in .env to enable AI chat routes.
"""
import os
import re
import pickle

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from flask import send_from_directory, request, jsonify

def _l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms

from models import db, AitaPost

USE_LLM = bool(os.getenv("SPARK_API_KEY"))

# Latent space size for truncated SVD. If this differs from data/index/svd_factors.npz
# on disk, factors are recomputed on next startup.
SVD_RANK = 20

_index = None  # loaded once from disk
_tfidf_cache = None

_SVD_NPZ = None  # path set after index is loaded


def _build_svd_dimension_labels(token_to_idx, words_svd_normed, terms_per_dimension=4):
    if words_svd_normed.size == 0:
        return []
    idx_to_token = [None] * len(token_to_idx)
    for token, idx in token_to_idx.items():
        if 0 <= idx < len(idx_to_token):
            idx_to_token[idx] = token

    labels = []
    for dim in range(words_svd_normed.shape[1]):
        dim_weights = words_svd_normed[:, dim]
        top_term_indices = np.argsort(np.abs(dim_weights))[::-1][:terms_per_dimension]
        top_terms = [idx_to_token[i] for i in top_term_indices if idx_to_token[i]]
        if top_terms:
            labels.append(", ".join(top_terms))
        else:
            labels.append(f"latent dimension {dim}")
    return labels


def _load_index():
    global _index, _SVD_NPZ
    if _index is not None:
        return _index
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_dir = os.path.join(project_root, 'data', 'index')
    meta_path = os.path.join(index_dir, 'tfidf_meta.pkl')
    npz_path  = os.path.join(index_dir, 'tfidf_matrix.npz')
    _SVD_NPZ  = os.path.join(index_dir, 'svd_factors.npz')
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
            return _l2_normalize_rows(docs), _l2_normalize_rows(words)
        if n_cols == 1:
            v = X_raw[:, 0].astype(np.float64, copy=False)
            nv = np.linalg.norm(v)
            docs = (v / nv).reshape(-1, 1) if nv > 0 else np.zeros((n_rows, 1))
            words = np.array([[1.0]], dtype=np.float64)
            return _l2_normalize_rows(docs), _l2_normalize_rows(words)
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
        return _l2_normalize_rows(docs_compressed), _l2_normalize_rows(words_compressed)

    # svds returns singular values in ascending order; flip to match largest-first latent dims
    docs_compressed = docs_compressed[:, ::-1]
    words_compressed = words_compressed[::-1, :].T
    return _l2_normalize_rows(docs_compressed), _l2_normalize_rows(words_compressed)


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
    global _tfidf_cache, _SVD_NPZ
    if _tfidf_cache is not None:
        return _tfidf_cache

    token_to_idx, idf, X, posts_meta = _load_index()
    X = X.tocsr()

    # Keep the precomputed TF-IDF matrix sparse. Converting it to a dense
    # array spikes memory into multi-GB territory and gets the Flask process
    # killed during the first search on many laptops.
    row_norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    X_normed = X.multiply(1.0 / row_norms[:, np.newaxis]).tocsr()

    # Load SVD from disk if rank matches; otherwise recompute (e.g. after SVD_RANK change).
    docs_svd_normed = None
    words_svd_normed = None
    if _SVD_NPZ and os.path.exists(_SVD_NPZ):
        saved = np.load(_SVD_NPZ)
        docs_svd_normed = saved['docs']
        words_svd_normed = saved['words']
        cached_rank = int(saved['rank'][0]) if 'rank' in saved.files else docs_svd_normed.shape[1]
        if docs_svd_normed.shape[1] != SVD_RANK or cached_rank != SVD_RANK:
            docs_svd_normed = None
            words_svd_normed = None

    if docs_svd_normed is None:
        docs_svd_normed, words_svd_normed = _build_svd(X.toarray(), k=SVD_RANK)
        if _SVD_NPZ:
            np.savez_compressed(
                _SVD_NPZ,
                docs=docs_svd_normed,
                words=words_svd_normed,
                rank=np.array([SVD_RANK], dtype=np.int32),
            )

    svd_dimension_labels = _build_svd_dimension_labels(token_to_idx, words_svd_normed)
    _tfidf_cache = (
        token_to_idx,
        idf,
        X_normed,
        docs_svd_normed,
        words_svd_normed,
        svd_dimension_labels,
        posts_meta,
    )
    return _tfidf_cache


def json_search(query, method='svd', verdict_filter=None):
    if not query or not query.strip():
        return []

    (
        token_to_idx,
        idf,
        X_normed,
        docs_svd_normed,
        words_svd_normed,
        svd_dimension_labels,
        posts,
    ) = _tfidf_index()
    if not posts or token_to_idx is None:
        return []
    if not token_to_idx or np.asarray(idf).size == 0:
        return []

    # Build candidate index set (all docs, or filtered by verdict)
    vf = verdict_filter.upper() if verdict_filter else None
    if vf:
        candidate_indices = np.array([
            i for i, p in enumerate(posts)
            if (p.get("verdict") if isinstance(p, dict) else getattr(p, "verdict", "UNKNOWN")) == vf
        ])
        if len(candidate_indices) == 0:
            return []
    else:
        candidate_indices = np.arange(len(posts))

    tokens = _tokenize(query)
    use_svd = method != 'tfidf'
    if not use_svd:
        q = _query_tfidf_l2(tokens, token_to_idx, idf)
        sims_all = X_normed @ q
    else:
        q = _query_svd(tokens, token_to_idx, idf, words_svd_normed)
        sims_all = docs_svd_normed @ q

    candidate_sims = sims_all[candidate_indices]
    top_local = np.argsort(candidate_sims)[::-1][:20]

    matches = []
    for local_idx in top_local:
        idx = int(candidate_indices[local_idx])
        post = posts[idx]
        svd_top_dimensions = None
        if use_svd:
            doc_vec = docs_svd_normed[idx]
            contributions = doc_vec * q
            top_dim_idx = np.argsort(np.abs(contributions))[::-1][:5]
            svd_top_dimensions = [
                {
                    "dimension": int(dim),
                    "label": (
                        svd_dimension_labels[int(dim)]
                        if int(dim) < len(svd_dimension_labels)
                        else f"latent dimension {int(dim)}"
                    ),
                    "post_value": float(doc_vec[dim]),
                    "query_value": float(q[dim]),
                    "contribution": float(contributions[dim]),
                }
                for dim in top_dim_idx
            ]

        if isinstance(post, dict):
            match = {
                "id": post.get("id"),
                "submission_id": post.get("submission_id"),
                "title": post.get("title"),
                "selftext": post.get("selftext"),
                "score": post.get("score"),
                "similarity": float(sims_all[idx]),
                "verdict": post.get("verdict", "UNKNOWN"),
            }
            if svd_top_dimensions is not None:
                match["svd_top_dimensions"] = svd_top_dimensions
            matches.append(match)
        else:
            match = {
                "id": post.id,
                "submission_id": post.submission_id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "similarity": float(sims_all[idx]),
                "verdict": getattr(post, "verdict", "UNKNOWN"),
            }
            if svd_top_dimensions is not None:
                match["svd_top_dimensions"] = svd_top_dimensions
            matches.append(match)
    return matches


def register_routes(app):
    @app.route('/', defaults={'path': ''}, methods=['GET'])
    @app.route('/<path:path>', methods=['GET'])
    def serve(path):
        if app.static_folder and path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        if app.static_folder and os.path.exists(os.path.join(app.static_folder, 'index.html')):
            return send_from_directory(app.static_folder, 'index.html')
        return jsonify({
            "message": "Frontend build not found. Run `npm run dev` in `frontend/` for local development or build the frontend first.",
            "use_llm": USE_LLM,
        }), 503

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/search")
    def search():
        query = request.args.get("query", "")
        method = request.args.get("method", "svd")
        verdict = request.args.get("verdict", None)
        return jsonify(json_search(query, method, verdict))

    if USE_LLM:
        from llm_routes import register_llm_search_route
        register_llm_search_route(app, json_search)
