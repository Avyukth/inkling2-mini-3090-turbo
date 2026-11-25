#!/usr/bin/env python3
"""Standalone TTFT, concurrency (c=1, 2, 4, 8), and Prometheus ms/step benchmark."""
import http.client
import json
import os
import re
import statistics
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get("INKLING_MODEL", "Inkling-2-Mini-Derisked")

BASE_TEXT = (
    "Distributed systems coordinate state across unreliable networks using consensus "
    "protocols, replicated logs, failure detectors, and bounded-staleness caches. "
)


def get_draft_metrics():
    txt = urllib.request.urlopen(f"http://{HOST}:{PORT}/metrics").read().decode()
    out = {}
    for k in ("num_drafts", "num_accepted_tokens"):
        m = re.findall(rf"^vllm:spec_decode_{k}_total\{{[^}}]*\}} ([0-9.e+]+)", txt, re.M)
        out[k] = sum(float(x) for x in m)
    return out


def stream_request(prompt, max_tokens=128, ignore_eos=False):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": ignore_eos,
        "reasoning_effort": "low",
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    ttft = None
    usage = {}
    buf = b""
    while True:
        data = resp.read(4096)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    conn.close()
    total = t1 - t0
    comp = usage.get("completion_tokens", 0)
    dec_s = max(1e-6, total - (ttft or total))
    return {
        "ttft_ms": (ttft or total) * 1000.0,
        "total_s": total,
        "prompt_toks": usage.get("prompt_tokens", 0),
        "comp_toks": comp,
        "decode_tps": (comp - 1) / dec_s if comp > 1 else 0.0,
    }


def run_concurrency_case(label, concurrency, approx_tokens, max_tokens=128, ignore_eos=False):
    reps = max(1, approx_tokens // 20)
    prompts = [
        f"[run-{uuid.uuid4().hex[:8]}] " + (BASE_TEXT * reps) + "\nSummarize in 3 bullet points."
        for _ in range(concurrency)
    ]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda p: stream_request(p, max_tokens=max_tokens, ignore_eos=ignore_eos), prompts))
    wall = time.perf_counter() - t0
    ttfts = sorted(r["ttft_ms"] for r in results)
    tot_comp = sum(r["comp_toks"] for r in results)
    print(
        f"{label:<22} | c={concurrency:<2} | ptok={int(statistics.mean(r['prompt_toks'] for r in results)):<5} | "
        f"TTFT avg={statistics.mean(ttfts):7.1f}ms max={ttfts[-1]:7.1f}ms | "
        f"stream={statistics.mean(r['decode_tps'] for r in results):5.1f} tok/s | "
        f"agg={tot_comp / wall:6.1f} tok/s"
    )


if __name__ == "__main__":
    # Wake GPU from idle P8 -> P2
    stream_request("hi", max_tokens=8)

    m0 = get_draft_metrics()
    r_code = stream_request(
        "Write a complete Python implementation of an LRU cache class with O(1) get/put, docstrings, and unit tests.",
        max_tokens=1500,
        ignore_eos=True,
    )
    m1 = get_draft_metrics()
    drafts = max(1.0, m1["num_drafts"] - m0["num_drafts"])
    print(
        f"Single-stream code (1500 tok, ignore_eos): {r_code['comp_toks'] / r_code['total_s']:.1f} tok/s | "
        f"{r_code['total_s'] * 1000.0 / drafts:.2f} ms/step"
    )
    print("-" * 105)
    for c in (1, 2, 4, 8):
        run_concurrency_case("cold ~210t", c, 128, max_tokens=128)
    for c in (1, 4, 8):
        run_concurrency_case("fixed-out ~210t", c, 128, max_tokens=256, ignore_eos=True)
    for c in (1, 2, 4):
        run_concurrency_case("cold ~3Kt", c, 2048, max_tokens=128)
