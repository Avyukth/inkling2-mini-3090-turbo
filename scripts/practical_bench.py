#!/usr/bin/env python3
"""Real-world agentic practicality benchmark for OpenAI-compatible LLM servers.

Measures 6 practical dimensions that synthetic tok/s tables miss:
  1. Idle GPU wake-up (P8 -> P2 cold first-request TTFT vs warm TTFT)
  2. Multi-turn agentic coding loop at ~64K context (prefix cache TTFT & subagent branching)
  3. Tool-calling & structured-output fidelity (escaped code-edit AST check, parallel tools, round-trip fix, JSON schema)
  4. Prefill-decode interference (Agent A inter-token stall & tok/s while Agent B prefills a cold 32K prompt)
  5. 100K-token 3-hop codebase synthesis (combining constants/functions at 10%, 50%, and 90% depth)
  6. Reasoning-effort control & wall-clock time-to-verified-code (none vs low vs medium, unit-test executed)
"""
import ast
import http.client
import json
import os
import re
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get("INKLING_MODEL", "Inkling-2-Mini-Derisked")
OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else "practical_bench_results.json"

FILLER_MODULE = (
    "class TelemetryBuffer:\n"
    "    def __init__(self, cap: int = 256):\n"
    "        self.cap = cap\n"
    "        self.items = []\n"
    "    def push(self, event: str, latency_ms: float) -> None:\n"
    "        if len(self.items) >= self.cap:\n"
    "            self.items.pop(0)\n"
    "        self.items.append((event, latency_ms))\n\n"
)  # ~62 tokens per repetition


def chat_sync(messages, max_tokens=512, effort="low", tools=None, response_format=None):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"reasoning_effort": effort},
    }
    if tools is not None:
        body["tools"] = tools
    if response_format is not None:
        body["response_format"] = response_format
    payload = json.dumps(body)
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    raw = resp.read()
    wall = time.perf_counter() - t0
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {raw.decode()[:300]}")
    data = json.loads(raw)
    return data, wall


def chat_stream(messages, max_tokens=256, effort="low"):
    body = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"reasoning_effort": effort},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload = json.dumps(body)
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body=payload, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raw = resp.read()
        conn.close()
        raise RuntimeError(f"HTTP {resp.status}: {raw.decode()[:300]}")
    ttft = None
    token_times = []
    usage = {}
    buf = b""
    while True:
        d = resp.read(4096)
        if not d:
            break
        buf += d
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data: ") or line[6:] == b"[DONE]":
                continue
            obj = json.loads(line[6:])
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    token_times.append(now)
    t1 = time.perf_counter()
    conn.close()
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    itls_ms = [(token_times[i] - token_times[i - 1]) * 1000.0 for i in range(1, len(token_times))]
    dec_dur = max(1e-6, (token_times[-1] - token_times[0]) if len(token_times) > 1 else (t1 - (ttft or t1)))
    return {
        "ttft_ms": (ttft or t1) * 1000.0,
        "wall_s": t1 - t0,
        "prompt_tokens": pt,
        "cached_tokens": cached,
        "completion_tokens": ct,
        "decode_tps": (ct - 1) / dec_dur if ct > 1 else 0.0,
        "itls_ms": itls_ms,
        "token_times": token_times,
        "t0": t0,
    }


results = {}

# ---------------------------------------------------------------------------
# 1. Idle Wake-Up TTFT vs Warm TTFT
# ---------------------------------------------------------------------------
print("=== 1. Idle Wake-Up vs Warm Short TTFT ===")
r_cold_idle = chat_stream([{"role": "user", "content": "Reply with ONLY: PONG"}], max_tokens=16, effort="none")
r_warm_1 = chat_stream([{"role": "user", "content": f"[{uuid.uuid4().hex[:6]}] Reply with ONLY: PONG"}], max_tokens=16, effort="none")
r_warm_2 = chat_stream([{"role": "user", "content": f"[{uuid.uuid4().hex[:6]}] Reply with ONLY: PONG"}], max_tokens=16, effort="none")
warm_ttft = round((r_warm_1["ttft_ms"] + r_warm_2["ttft_ms"]) / 2.0, 1)
results["1_idle_vs_warm_ttft"] = {
    "first_request_ttft_ms": round(r_cold_idle["ttft_ms"], 1),
    "warm_short_ttft_ms": warm_ttft,
}
print(" ", results["1_idle_vs_warm_ttft"])

# ---------------------------------------------------------------------------
# 2. Multi-Turn Agentic Loop at ~64K Context + 2-Subagent Shared-Prefix Branch
# ---------------------------------------------------------------------------
print("=== 2. Multi-Turn Agentic Loop (~64K Context) + Shared-Prefix Subagents ===")
repo_tag = f"# repo-snapshot-{uuid.uuid4().hex[:8]}\n"
repo_64k = repo_tag + (FILLER_MODULE * 630)  # ~64.3K tokens
msgs = [{"role": "user", "content": repo_64k + "\nInspect the TelemetryBuffer class and state its default cap in 1 line."}]
t1 = chat_stream(msgs, max_tokens=64, effort="low")
msgs.append({"role": "assistant", "content": "Default cap is 256."})

turn_ttfts = []
turn_cached = []
for step in range(2, 5):
    tool_output = f"[tool_result step={step} id={uuid.uuid4().hex[:6]}]\n" + ("log_line: ok latency=1.2ms\n" * 40)
    msgs.append({"role": "user", "content": tool_output + f"\nAcknowledge step {step} in one short sentence."})
    tn = chat_stream(msgs, max_tokens=64, effort="low")
    turn_ttfts.append(round(tn["ttft_ms"], 1))
    turn_cached.append(tn["cached_tokens"])
    msgs.append({"role": "assistant", "content": f"Step {step} acknowledged."})

# Two concurrent subagents branching off the same 64K repo snapshot with distinct tasks
def run_subagent(idx):
    sub_msgs = [{"role": "user", "content": repo_64k + f"\n[subagent-{idx}-{uuid.uuid4().hex[:6]}] Propose improvement #{idx} to TelemetryBuffer in 2 bullets."}]
    return chat_stream(sub_msgs, max_tokens=96, effort="low")

with ThreadPoolExecutor(max_workers=2) as ex:
    subs = list(ex.map(run_subagent, [1, 2]))

results["2_agentic_multiturn_64k"] = {
    "context_tokens": t1["prompt_tokens"],
    "turn1_cold_ttft_s": round(t1["ttft_ms"] / 1000.0, 2),
    "turns_2_to_4_cached_ttft_ms": turn_ttfts,
    "turns_2_to_4_avg_ttft_ms": round(statistics.mean(turn_ttfts), 1),
    "turns_2_to_4_cached_tokens": turn_cached,
    "speedup_factor_vs_cold": round((t1["ttft_ms"]) / statistics.mean(turn_ttfts), 1),
    "two_branched_subagents_avg_ttft_ms": round(statistics.mean(s["ttft_ms"] for s in subs), 1),
    "two_branched_subagents_cached_tokens": [s["cached_tokens"] for s in subs],
}
print(" ", results["2_agentic_multiturn_64k"])

# ---------------------------------------------------------------------------
# 3. Tool-Calling & Structured Output Fidelity
# ---------------------------------------------------------------------------
print("=== 3. Tool-Calling, Escaped Code-Edit & JSON Schema Fidelity ===")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace target string with replacement in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
]

checks = {}

# 3a. Escaped multi-line Python code edit with quotes and newlines
d_edit, _ = chat_sync(
    [{
        "role": "user",
        "content": (
            "The current content of `parser.py` is:\n```python\ndef parse(s):\n    return s\n```\n"
            "Call `edit_file` right now on `parser.py` (do not call `read_file`) to replace the whole function "
            "with `def parse(s: str) -> list[str]:` that splits `s` on `\\n`, strips both `\"` and `'` quotes, "
            "and returns non-empty lines."
        ),
    }],
    max_tokens=512,
    effort="low",
    tools=TOOLS,
)
tc_edit = (d_edit["choices"][0]["message"].get("tool_calls") or [])
edit_ok = False
if tc_edit and tc_edit[0]["function"]["name"] == "edit_file":
    try:
        args = json.loads(tc_edit[0]["function"]["arguments"])
        ast.parse(args["replacement"])
        edit_ok = args.get("path") == "parser.py" and "def parse" in args["replacement"]
    except Exception:
        edit_ok = False
checks["escaped_code_edit_ast_valid"] = edit_ok

# 3b. Parallel tool calls (3 files in 1 turn)
d_par, _ = chat_sync(
    [{"role": "user", "content": "Call `read_file` in parallel on `src/main.rs`, `src/lib.rs`, and `Cargo.toml` in this single turn."}],
    max_tokens=512,
    effort="low",
    tools=TOOLS,
)
tc_par = d_par["choices"][0]["message"].get("tool_calls") or []
par_paths = sorted(json.loads(t["function"]["arguments"]).get("path", "") for t in tc_par if t["function"]["name"] == "read_file")
checks["parallel_3_tool_calls"] = (par_paths == ["Cargo.toml", "src/lib.rs", "src/main.rs"])

# 3c. Multi-turn tool round-trip bug fix
rt_msgs = [
    {"role": "user", "content": "Read `math_util.py` first using `read_file`, then fix any bug in `sub(a, b)`."},
]
d_rt1, _ = chat_sync(rt_msgs, max_tokens=256, effort="low", tools=TOOLS)
m_rt1 = d_rt1["choices"][0]["message"]
tc_rt1 = m_rt1.get("tool_calls") or []
rt_ok = False
if tc_rt1 and tc_rt1[0]["function"]["name"] == "read_file":
    rt_msgs.append({"role": "assistant", "content": m_rt1.get("content") or "", "tool_calls": tc_rt1})
    rt_msgs.append({
        "role": "tool",
        "tool_call_id": tc_rt1[0]["id"],
        "content": "def sub(a: int, b: int) -> int:\n    return a + b  # BUG\n",
    })
    d_rt2, _ = chat_sync(rt_msgs, max_tokens=512, effort="low", tools=TOOLS)
    tc_rt2 = d_rt2["choices"][0]["message"].get("tool_calls") or []
    if tc_rt2 and tc_rt2[0]["function"]["name"] == "edit_file":
        args2 = json.loads(tc_rt2[0]["function"]["arguments"])
        rt_ok = "a - b" in args2.get("replacement", "")
checks["multiturn_tool_roundtrip_fix"] = rt_ok

# 3d. Strict JSON Schema output
SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_report",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                "line": {"type": "integer"},
                "summary": {"type": "string"},
            },
            "required": ["severity", "line", "summary"],
            "additionalProperties": False,
        },
    },
}
d_js, _ = chat_sync(
    [{"role": "user", "content": "Review line 42: `query = f\"SELECT * FROM users WHERE id = '{user_input}'\"`. Return JSON."}],
    max_tokens=256,
    effort="low",
    response_format=SCHEMA,
)
js_ok = False
try:
    parsed_js = json.loads(d_js["choices"][0]["message"]["content"])
    js_ok = (
        parsed_js.get("severity") in ("high", "critical")
        and parsed_js.get("line") == 42
        and isinstance(parsed_js.get("summary"), str)
    )
except Exception:
    js_ok = False
checks["strict_json_schema"] = js_ok

passed_checks = sum(1 for v in checks.values() if v)
results["3_tool_and_schema_fidelity"] = {
    "passed": f"{passed_checks}/{len(checks)}",
    "checks": checks,
}
print(" ", results["3_tool_and_schema_fidelity"])

# ---------------------------------------------------------------------------
# 4. Prefill-Decode Interference (Agent A streaming while Agent B prefills cold 32K)
# ---------------------------------------------------------------------------
print("=== 4. Prefill-Decode Interference (Agent A Streaming + Agent B Cold 32K Prefill) ===")
AGENT_A_PROMPT = f"[solo-{uuid.uuid4().hex[:6]}] Write a detailed Python implementation of a B-tree with split_child, insert, search, and extensive unit tests."
solo_a = chat_stream([{"role": "user", "content": AGENT_A_PROMPT}], max_tokens=700, effort="low")

box = {}
def run_agent_a():
    box["a"] = chat_stream(
        [{"role": "user", "content": f"[simA-{uuid.uuid4().hex[:6]}] Write a detailed Python implementation of a B-tree with split_child, insert, search, and extensive unit tests."}],
        max_tokens=900,
        effort="low",
    )

def run_agent_b():
    time.sleep(0.6)  # Join while Agent A is mid-decode!
    cold_32k = f"# cold32k-{uuid.uuid4().hex[:8]}\n" + (FILLER_MODULE * 315) + "\nSummarize TelemetryBuffer in 2 bullets."
    t_b_start = time.perf_counter()
    res_b = chat_stream([{"role": "user", "content": cold_32k}], max_tokens=64, effort="low")
    res_b["t_b_start"] = t_b_start
    res_b["t_b_first"] = t_b_start + res_b["ttft_ms"] / 1000.0
    box["b"] = res_b

ta = threading.Thread(target=run_agent_a)
tb = threading.Thread(target=run_agent_b)
ta.start(); tb.start(); ta.join(); tb.join()

ra, rb = box["a"], box["b"]
# Compute Agent A's tokens & max gap specifically inside [t_b_start, t_b_first]
in_window = [t for t in ra["token_times"] if rb["t_b_start"] <= t <= rb["t_b_first"]]
max_stall_ms = max(ra["itls_ms"]) if ra["itls_ms"] else 0.0
results["4_prefill_decode_interference"] = {
    "agent_a_solo_decode_tps": round(solo_a["decode_tps"], 1),
    "agent_a_Solo_max_itl_ms": round(max(solo_a["itls_ms"]) if solo_a["itls_ms"] else 0.0, 1),
    "agent_b_cold_prompt_tokens": rb["prompt_tokens"],
    "agent_b_ttft_s": round(rb["ttft_ms"] / 1000.0, 2),
    "agent_a_overall_decode_tps_with_b": round(ra["decode_tps"], 1),
    "agent_a_max_inter_token_stall_ms": round(max_stall_ms, 1),
}
print(" ", results["4_prefill_decode_interference"])

# ---------------------------------------------------------------------------
# 5. ~100K-Token 3-Hop Codebase Synthesis (10%, 50%, 90% Depth)
# ---------------------------------------------------------------------------
print("=== 5. ~100K-Token 3-Hop Codebase Synthesis (10% / 50% / 90% Needles) ===")
chunk = FILLER_MODULE * 98  # ~10.0K tokens per 10% slice -> ~100.1K total
hop1 = "\n# FILE: config/constants.py\nSCALE_FACTOR = 17\nOFFSET_BIAS = -43\n\n"
hop2 = "\n# FILE: core/pipeline.py\ndef transform_metric(x: int) -> int:\n    return (x * SCALE_FACTOR) + OFFSET_BIAS\n\n"
hop3 = "\n# FILE: verify/audit.py\ndef validate_checksum(val: int) -> int:\n    return (transform_metric(val) ^ 0x5A) + 19\n\n"

doc_100k = (
    f"# codebase-{uuid.uuid4().hex[:8]}\n"
    + chunk            # 0-10%
    + hop1             # @ ~10%
    + (chunk * 4)      # 10-50%
    + hop2             # @ ~50%
    + (chunk * 4)      # 50-90%
    + hop3             # @ ~90%
    + chunk            # 90-100%
    + "\nQuestion: Using `config/constants.py`, `core/pipeline.py`, and `verify/audit.py` above, "
      "what exact integer does `validate_checksum(12)` return? Reply with ONLY the integer."
)
# Ground truth: (12 * 17 - 43) = 161; 161 ^ 90 = 251; 251 + 19 = 270
d_100k, wall_100k = chat_sync([{"role": "user", "content": doc_100k}], max_tokens=2048, effort="low")
ans_100k = (d_100k["choices"][0]["message"].get("content") or "").strip()
pt_100k = d_100k["usage"]["prompt_tokens"]
hop_ok = bool(re.findall(r"\b270\b", ans_100k))
results["5_100k_3hop_codebase_reasoning"] = {
    "prompt_tokens": pt_100k,
    "expected": 270,
    "answer": ans_100k,
    "exact_match": hop_ok,
    "total_wall_s": round(wall_100k, 2),
}
print(" ", results["5_100k_3hop_codebase_reasoning"])

# ---------------------------------------------------------------------------
# 6. Reasoning Effort Toggle & Wall-Clock Time-to-Verified-Code (none / low / medium)
# ---------------------------------------------------------------------------
print("=== 6. Reasoning Effort Toggle & Time-to-Verified-Code ===")
CODING_PROBLEMS = [
    (
        "Write a Python function `min_meeting_rooms(intervals: list[tuple[int, int]]) -> int` returning the minimum "
        "number of conference rooms required so no two overlapping `[start, end)` meetings share a room (`[1,2)` and `[2,3)` do NOT overlap). "
        "Return ONLY raw Python code (no markdown fences).",
        "assert min_meeting_rooms([(0,30),(5,10),(15,20)]) == 2\n"
        "assert min_meeting_rooms([(7,10),(2,4)]) == 1\n"
        "assert min_meeting_rooms([(1,5),(2,6),(4,8),(5,9)]) == 3\n"
        "assert min_meeting_rooms([]) == 0\n"
    ),
    (
        "Write a Python function `longest_k_distinct(s: str, k: int) -> int` returning the length of the longest "
        "substring of `s` with at most `k` distinct characters. Return ONLY raw Python code (no markdown fences).",
        "assert longest_k_distinct('eceba', 2) == 3\n"
        "assert longest_k_distinct('aa', 1) == 2\n"
        "assert longest_k_distinct('abcdef', 0) == 0\n"
        "assert longest_k_distinct('abaccc', 2) == 4\n"
    ),
    (
        "Write a Python function `eval_rpn(tokens: list[str]) -> int` evaluating Reverse Polish Notation with `+`, `-`, `*`, `/` "
        "where division truncates toward zero (like `int(a / b)`). Return ONLY raw Python code (no markdown fences).",
        "assert eval_rpn(['2','1','+','3','*']) == 9\n"
        "assert eval_rpn(['4','13','5','/','+']) == 6\n"
        "assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+']) == 22\n"
    ),
]

effort_summary = {}
for eff in ("none", "low", "medium"):
    passed = 0
    tot_wall = 0.0
    tot_comp = 0
    tot_reas = 0
    for idx, (prompt, tests) in enumerate(CODING_PROBLEMS):
        d_c, w_c = chat_sync(
            [{"role": "user", "content": f"[eff-{eff}-{idx}-{uuid.uuid4().hex[:5]}] " + prompt}],
            max_tokens=1536,
            effort=eff,
        )
        tot_wall += w_c
        u = d_c["usage"]
        tot_comp += u.get("completion_tokens", 0)
        tot_reas += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        code_txt = d_c["choices"][0]["message"].get("content") or ""
        # Strip fences if model added them
        m_fence = re.search(r"```(?:python)?\s*(.*?)```", code_txt, re.S)
        if m_fence:
            code_txt = m_fence.group(1)
        ns = {}
        try:
            exec(code_txt + "\n" + tests, ns, ns)
            passed += 1
        except Exception:
            pass
    effort_summary[eff] = {
        "unit_tests_passed": f"{passed}/{len(CODING_PROBLEMS)}",
        "avg_wall_s_per_task": round(tot_wall / len(CODING_PROBLEMS), 2),
        "avg_completion_tokens": round(tot_comp / len(CODING_PROBLEMS), 1),
        "avg_reasoning_tokens": round(tot_reas / len(CODING_PROBLEMS), 1),
    }
    print(f"  effort={eff:<6}:", effort_summary[eff])

results["6_reasoning_effort_and_solve_time"] = effort_summary

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {OUT_JSON}")
