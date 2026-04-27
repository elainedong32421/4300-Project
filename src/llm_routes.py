"""
LLM routes — only loaded when USE_LLM = True in routes.py.

Registers two endpoints:
  POST /api/llm_search  — non-streaming RAG: rewrite → IR → LLM rerank → verdict synthesis
  POST /api/rag         — streaming RAG: same pipeline but SSE so UI can show
                          IR results before the LLM answer finishes
"""
import json
import re
import os
import logging
import socket
from flask import request, jsonify, Response, stream_with_context
from infosci_spark_client import LLMClient

logger = logging.getLogger(__name__)

_REWRITE_SYSTEM = (
    "You are a search query optimizer for a Reddit AITA post database. "
    "The user will describe a personal situation. Rewrite it as a short keyword-rich query "
    "(max 15 words) that would surface similar AITA posts. Strip emotional language. "
    "Focus on the core conflict and key roles (e.g., roommate, wedding, money, family). "
    "Return ONLY the rewritten query, nothing else."
)

_SYNTHESIS_SYSTEM = (
    "You are an AITA verdict assistant. You are given a user's situation and a ranked list of "
    "similar Reddit AITA posts retrieved from the database — ordered from most to least relevant. "
    "Structure your response in exactly this format:\n\n"
    "**Verdict: [NTA / YTA / ESH / NAH]**\n"
    "[2-3 sentences explaining why. You MUST directly cite at least 2 of the retrieved posts by "
    "referring to their title or specific situation (e.g. 'In Post #1, where someone refused to pay "
    "for a sibling's wedding, the verdict was NTA because...'). "
    "Draw your reasoning from what those specific posts show. "
    "Be direct and match Reddit's tone. Do NOT invent posts not in the list.]\n\n"
    "**Example prompt:** \"[Write a clear, specific version of the user's situation reworded "
    "as a good AITA-style question — third-person, concrete details, under 30 words — "
    "that would retrieve even better results from the database.]\""
)

_LLM_RERANK_SYSTEM = (
    "You re-rank Reddit AITA posts by relevance to a user's specific situation. "
    "Given the user's query and up to 10 retrieved posts in their current order, "
    "re-order them from most to least relevant. "
    "You may keep the original order unchanged if it is already optimal. "
    "For each post write ONE sentence (max 12 words) explaining why it is at that position. "
    "Respond with ONLY valid JSON — an array ordered by new rank: "
    "[{\"original_rank\": N, \"reason\": \"...\"}, ...] "
    "Include every post. original_rank is 1-based."
)


def _make_client():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        return None, "SPARK_API_KEY not set in .env"
    socket.setdefaulttimeout(60)
    return LLMClient(api_key=api_key.strip()), None


def _llm_rerank(client, user_query, posts):
    """
    Ask the LLM to re-rank posts by semantic relevance to the user's situation.
    Returns posts in new order with original_rank and rerank_reason added.
    Falls back to original order with empty reasons on any error.
    """
    lines = "\n".join(
        f"Post {i+1} [{p.get('verdict', 'UNKNOWN')}]: {p.get('title', '')}"
        for i, p in enumerate(posts)
    )
    prompt = (
        f"User's situation: {user_query}\n\n"
        f"Retrieved posts (current order):\n{lines}\n\n"
        "Re-rank these posts by relevance to the user's specific situation. "
        "Respond with ONLY a valid JSON array."
    )
    try:
        resp = client.chat([
            {"role": "system", "content": _LLM_RERANK_SYSTEM},
            {"role": "user", "content": prompt},
        ])
        raw = (resp.get("content") or "").strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        items = json.loads(raw)

        post_by_rank = {i + 1: p for i, p in enumerate(posts)}
        reranked = []
        seen = set()
        for item in items:
            orig_rank = int(item.get("original_rank", 0))
            if orig_rank in post_by_rank and orig_rank not in seen:
                seen.add(orig_rank)
                reranked.append({
                    **post_by_rank[orig_rank],
                    'original_rank': orig_rank,
                    'rerank_reason': item.get("reason", ""),
                })
        # Safety net: append any posts the LLM omitted
        for i, p in enumerate(posts):
            if (i + 1) not in seen:
                reranked.append({**p, 'original_rank': i + 1, 'rerank_reason': ''})
        return reranked
    except Exception as e:
        logger.warning(f"LLM rerank error: {e}")
        return [{**p, 'original_rank': i + 1, 'rerank_reason': ''} for i, p in enumerate(posts)]


def register_llm_search_route(app, json_search):
    """Register /api/llm_search and /api/rag endpoints."""

    @app.route("/api/llm_search", methods=["POST"])
    def llm_search():
        """Non-streaming RAG — kept for compatibility."""
        data = request.get_json() or {}
        user_query = (data.get("query") or "").strip()
        verdict_filter = data.get("verdict_filter") or None
        method = (data.get("method") or "svd").lower()

        if not user_query:
            return jsonify({"error": "query is required"}), 400

        client, err = _make_client()
        if err:
            return jsonify({"error": err}), 500

        # Step 1 — Query rewriting
        rewrite_resp = client.chat([
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": user_query},
        ])
        rewritten_query = (rewrite_resp.get("content") or user_query).strip()

        # Step 2 — IR retrieval
        ir_results = json_search(rewritten_query, method=method, verdict_filter=verdict_filter)

        # Step 3 — LLM re-rank top 10 by semantic relevance
        reranked = _llm_rerank(client, user_query, ir_results[:10])

        # Step 4 — Verdict synthesis (use reranked order so most relevant posts are cited first)
        context_posts = reranked if reranked else ir_results[:10]
        posts_context = "\n\n".join(
            f"Post #{i+1} [{r.get('verdict', 'UNKNOWN')}]: {r['title']}\n{(r.get('selftext') or '')[:500]}"
            for i, r in enumerate(context_posts)
        ) or "No relevant posts found."

        synthesis_resp = client.chat([
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": (
                f"User's situation:\n{user_query}\n\n"
                f"Retrieved AITA posts (ranked by relevance):\n{posts_context}"
            )},
        ])
        llm_answer = (synthesis_resp.get("content") or "").strip()

        return jsonify({
            "rewritten_query": rewritten_query,
            "ir_results": ir_results,
            "reranked_results": reranked,
            "llm_answer": llm_answer,
            "verdict_filter": verdict_filter,
        })

    @app.route("/api/rag", methods=["POST"])
    def rag():
        """
        Streaming RAG pipeline over SSE.

        Events emitted in order:
          1. {"rewritten_query": "..."}          — LLM-rewritten IR query
          2. {"ir_results": [...]}                — retrieved posts (original SVD/TF-IDF ranking)
          3. {"reranked_results": [...]}           — TF-IDF re-ranked with 1-line LLM reasons
          4. {"content": "..."}  (many)           — streaming LLM synthesis tokens
          5. {"done": true}                       — stream finished
        """
        data = request.get_json() or {}
        user_query = (data.get("query") or "").strip()
        method = (data.get("method") or "svd").lower()

        if not user_query:
            return jsonify({"error": "query is required"}), 400

        client, err = _make_client()
        if err:
            return jsonify({"error": err}), 500

        def generate():
            # Step 1 — rewrite query
            try:
                rewrite_resp = client.chat([
                    {"role": "system", "content": _REWRITE_SYSTEM},
                    {"role": "user", "content": user_query},
                ])
                rewritten_query = (rewrite_resp.get("content") or user_query).strip()
            except Exception as e:
                logger.error(f"Rewrite error: {e}")
                yield f"data: {json.dumps({'error': f'LLM error: {e}'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            yield f"data: {json.dumps({'rewritten_query': rewritten_query})}\n\n"

            # Step 2 — IR retrieval
            try:
                ir_results = json_search(rewritten_query, method=method)
            except Exception as e:
                logger.error(f"IR error: {e}")
                ir_results = []

            yield f"data: {json.dumps({'ir_results': ir_results})}\n\n"

            if not ir_results:
                yield f"data: {json.dumps({'content': 'No relevant posts found in the database.'})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                return

            # Step 3 — LLM re-rank top 10 by semantic relevance
            try:
                reranked = _llm_rerank(client, user_query, ir_results[:10])
            except Exception as e:
                logger.error(f"Rerank error: {e}")
                reranked = []

            yield f"data: {json.dumps({'reranked_results': reranked})}\n\n"

            # Step 4 — synthesize verdict using reranked order so most relevant posts appear first
            context_posts = reranked if reranked else ir_results[:10]
            posts_context = "\n\n".join(
                f"Post #{i+1} [{r.get('verdict', 'UNKNOWN')}]: {r['title']}\n{(r.get('selftext') or '')[:500]}"
                for i, r in enumerate(context_posts)
            )
            synthesis_messages = [
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user", "content": (
                    f"User's situation:\n{user_query}\n\n"
                    f"Retrieved AITA posts (ranked by relevance):\n{posts_context}"
                )},
            ]

            got_content = False
            try:
                for chunk in client.chat(synthesis_messages, stream=True):
                    if chunk.get("content"):
                        got_content = True
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming synthesis error: {e}")

            if not got_content:
                try:
                    resp = client.chat(synthesis_messages)
                    answer = (resp.get("content") or "").strip()
                    if answer:
                        yield f"data: {json.dumps({'content': answer})}\n\n"
                    else:
                        yield f"data: {json.dumps({'error': 'LLM returned no content. Check SPARK_API_KEY.'})}\n\n"
                except Exception as e:
                    logger.error(f"Blocking synthesis error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
