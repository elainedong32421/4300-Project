"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
import re

import numpy as np
from flask import send_from_directory, request, jsonify
#from streamlit import text

from models import db, AitaPost 

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

# Cache: (episode_count, token_to_idx, idf, X_l2_rows, pairs)
_tfidf_cache = None


def _post_text(post):
    return f"{post.title} {post.selftext}"


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


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
    X = tf * idf[np.newaxis, :]

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    X = X / norms
    return token_to_idx, idf, X


def _query_tfidf_l2(tokenized_q, token_to_idx, idf):
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
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    return q


def _tfidf_index():
    global _tfidf_cache
    n = AitaPost.query.count()
    if _tfidf_cache is not None and _tfidf_cache[0] == n:
        return _tfidf_cache[1], _tfidf_cache[2], _tfidf_cache[3], _tfidf_cache[4]

    posts = AitaPost.query.all()

    if not posts:
        _tfidf_cache = (0, None, None, None, [])
        return None, None, None, []

    tokenized = [_tokenize(_post_text(post)) for post in posts]
    token_to_idx, idf, X = _build_tfidf_l2_rows(tokenized)
    _tfidf_cache = (n, token_to_idx, idf, X, posts)
    return token_to_idx, idf, X, posts


def json_search(query):
    if not query or not query.strip():
        return []

    token_to_idx, idf, X, posts = _tfidf_index()
    if not posts or token_to_idx is None or idf.size == 0:
        return []

    q = _query_tfidf_l2(_tokenize(query), token_to_idx, idf)
    sims = X @ q
    order = np.argsort(sims)[::-1]

    matches = []
    for idx in order[:20]:
        post = posts[int(idx)]
        matches.append(
            {
                "id": post.id,
                "submission_id": post.submission_id,
                "title": post.title,
                "selftext": post.selftext,
                "score": post.score,
                "similarity": float(sims[int(idx)]),
            }
        )
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
        text = request.args.get("query", "")
        return jsonify(json_search(text))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
