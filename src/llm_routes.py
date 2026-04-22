"""
LLM routes — only loaded when USE_LLM = True in routes.py.

Registers two endpoints:
  POST /api/llm_search  — non-streaming RAG: rewrite → IR → verdict synthesis
  POST /api/rag         — streaming RAG: same pipeline but SSE so UI can show
                          IR results before the LLM answer finishes
"""
import json
import os
import logging
import socket
import requests
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
    "You are an AITA verdict assistant. You are given a user's situation and retrieved similar "
    "Reddit AITA posts from the database. Structure your response in exactly this format:\n\n"
    "**Verdict: [NTA / YTA / ESH / NAH]**\n"
    "[2-3 sentences explaining why, referencing patterns from the retrieved posts. "
    "Be direct and match Reddit's tone. Do NOT invent posts.]\n\n"
    "**Example prompt:** \"[Write a clear, specific version of the user's situation reworded "
    "as a good AITA-style question — third-person, concrete details, under 30 words — "
    "that would retrieve even better results from the database.]\""
)


def _make_client():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        return None, "SPARK_API_KEY not set in .env"
    # Set a global socket timeout so LLM calls never hang indefinitely
    socket.setdefaulttimeout(60)
    return LLMClient(api_key=api_key.strip()), None


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

        # Step 3 — Verdict synthesis
        posts_context = "\n\n".join(
            f"Post {i+1} [{r.get('verdict', 'UNKNOWN')}]: {r['title']}\n{(r.get('selftext') or '')[:300]}"
            for i, r in enumerate(ir_results)
        ) or "No relevant posts found."

        synthesis_resp = client.chat([
            {"role": "system", "content": _SYNTHESIS_SYSTEM},
            {"role": "user", "content": (
                f"User's situation:\n{user_query}\n\n"
                f"Retrieved AITA posts:\n{posts_context}"
            )},
        ])
        llm_answer = (synthesis_resp.get("content") or "").strip()

        return jsonify({
            "rewritten_query": rewritten_query,
            "ir_results": ir_results,
            "llm_answer": llm_answer,
            "verdict_filter": verdict_filter,
        })

    @app.route("/api/rag", methods=["POST"])
    def rag():
        """
        Streaming RAG pipeline over SSE.

        Events emitted in order:
          1. {"rewritten_query": "..."}          — LLM-rewritten IR query
          2. {"ir_results": [...]}                — retrieved posts (same schema as /api/search)
          3. {"content": "..."}  (many)           — streaming LLM synthesis tokens
          4. {"done": true}                       — stream finished
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

            # Step 3 — synthesize verdict (try streaming, fall back to blocking)
            posts_context = "\n\n".join(
                f"Post {i+1} [{r.get('verdict', 'UNKNOWN')}]: {r['title']}\n{(r.get('selftext') or '')[:300]}"
                for i, r in enumerate(ir_results)
            )
            synthesis_messages = [
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user", "content": (
                    f"User's situation:\n{user_query}\n\n"
                    f"Retrieved AITA posts:\n{posts_context}"
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

            # Streaming returned nothing — try blocking call instead
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
