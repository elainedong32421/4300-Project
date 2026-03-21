"""
Routes: React app serving and episode search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
import re
import pickle

import numpy as np
import scipy.sparse as sp
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


def _query_vec(tokens, token_to_idx, idf):
    V = len(idf)
    q = np.zeros(V, dtype=np.float32)
    for t in tokens:
        j = token_to_idx.get(t)
        if j is not None:
            q[j] += 1.0
    mask = q > 0
    q[mask] = (1.0 + np.log(q[mask])) * idf[mask]
    n = np.linalg.norm(q)
    if n > 0:
        q /= n
    return q


def json_search(query):
    if not query or not query.strip():
        return []

    token_to_idx, idf, X, posts = _load_index()
    q = _query_vec(_tokenize(query), token_to_idx, idf)
    sims = X.dot(q)
    order = np.argsort(sims)[::-1]

    matches = []
    for idx in order[:20]:
        post = posts[int(idx)]
        matches.append({
            "id": post['id'],
            "submission_id": post['submission_id'],
            "title": post['title'],
            "selftext": post['selftext'],
            "score": post['score'],
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
        text = request.args.get("query", "")
        return jsonify(json_search(text))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
