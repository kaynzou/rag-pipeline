#!/usr/bin/env python3
"""
Verify a deployed voice-enabled RAG pipeline.

Usage:
    python verify_deployment.py --api-url https://<your-api-url> [--audio test.wav] [--frontend-url https://<your-frontend-url>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def request(method: str, url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        return e.code, body
    except Exception as e:
        return None, str(e)


def check(name: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    parser = argparse.ArgumentParser(description="Verify deployed voice-enabled RAG pipeline")
    parser.add_argument("--api-url", required=True, help="Base API URL, e.g. https://rag-pipeline-1.onrender.com")
    parser.add_argument("--audio", default=None, help="Path to a WAV/MP3 file for voice query test")
    parser.add_argument("--frontend-url", default=None, help="Frontend URL to check HTTP 200")
    parser.add_argument("--query-count", type=int, default=3, help="Number of text queries to run for latency stats")
    args = parser.parse_args()

    api = args.api_url.rstrip("/")
    passed = 0
    failed = 0

    print("=" * 70)
    print(f"Verifying: {api}")
    print("=" * 70)

    # Health
    status, body = request("GET", f"{api}/health")
    ok = check("GET /health", status == 200 and '"ok"' in body, body[:120] if body else "")
    passed += ok; failed += not ok

    # Ready
    status, body = request("GET", f"{api}/ready")
    if status == 200:
        data = json.loads(body)
        ready = data.get("ready", False)
        indexed = data.get("indexed", False)
        ok = check("GET /ready", ready and indexed, f"ready={ready}, indexed={indexed}")
    else:
        ok = check("GET /ready", False, f"status={status}")
    passed += ok; failed += not ok

    if not ok:
        print("\nPipeline not ready. Index corpus first with:")
        print(f"  curl -X POST {api}/index -H 'Content-Type: application/json' -d '{{\"source\":\"sample_corpus\"}}'")
        sys.exit(1)

    # Text queries
    questions = [
        "What is BM25?",
        "How does RAG work?",
        "What is a cross-encoder reranker?",
    ]
    latencies = []
    for q in questions[: args.query_count]:
        payload = json.dumps({"question": q, "top_k": 3}).encode("utf-8")
        status, body = request("POST", f"{api}/query", data=payload, headers={"Content-Type": "application/json"})
        if status == 200:
            data = json.loads(body)
            latency = data.get("latency_ms", 0)
            latencies.append(latency)
            guard = data.get("guardrail_passed")
            ok = check(
                f"POST /query: {q[:40]}",
                "answer" in data and len(data.get("sources", [])) > 0 and guard is not None,
                f"latency={latency:.1f}ms, guardrail={guard}, chunks_retrieved={data.get('chunks_retrieved')}",
            )
        else:
            ok = check(f"POST /query: {q[:40]}", False, f"status={status}, body={body[:120]}")
        passed += ok; failed += not ok

    # Latency stats
    status, body = request("GET", f"{api}/latency")
    if status == 200:
        data = json.loads(body)
        p50 = data.get("p50_ms", 0)
        p70 = data.get("p70_ms", 0)
        p100 = data.get("p100_ms", 0)
        n = data.get("num_queries", 0)
        ok = check("GET /latency", p50 > 0 and n > 0, f"P50={p50:.1f}ms, P70={p70:.1f}ms, P100={p100:.1f}ms, queries={n}")
    else:
        ok = check("GET /latency", False, f"status={status}")
    passed += ok; failed += not ok

    # Voice query
    if args.audio and Path(args.audio).exists():
        boundary = "----FormBoundary7MA4YWxkTrZu0gW"
        with open(args.audio, "rb") as f:
            audio_bytes = f.read()
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{Path(args.audio).name}"\r\n'
            f"Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8") + audio_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")
        status, resp = request("POST", f"{api}/voice/query?top_k=3", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        if status == 200:
            data = json.loads(resp)
            ok = check(
                "POST /voice/query",
                "transcript" in data and "answer" in data and data.get("stt_latency_ms", 0) > 0,
                f"transcript={data.get('transcript','')[:60]}, stt_latency={data.get('stt_latency_ms',0):.1f}ms, total={data.get('latency_ms',0):.1f}ms",
            )
        else:
            ok = check("POST /voice/query", False, f"status={status}, body={resp[:180]}")
        passed += ok; failed += not ok
    else:
        print("[ SKIP] POST /voice/query (pass --audio test.wav to enable)")

    # Frontend
    if args.frontend_url:
        status, body = request("GET", args.frontend_url)
        ok = check("Frontend GET /", status == 200, f"status={status}")
        passed += ok; failed += not ok
    else:
        print("[ SKIP] Frontend check (pass --frontend-url https://... to enable)")

    # Summary
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
