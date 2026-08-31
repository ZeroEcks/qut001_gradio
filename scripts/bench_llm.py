#!/usr/bin/env python3
"""Load-test apps/llm_demo.py's /on_start endpoint with dummy prompts.

Talks directly to a plain URL over the Gradio queue protocol (no dependency
on hf-perftest's internals), so it works against anything already listening
there: a local `python apps/llm_demo.py`, `docker compose up`, or a
`kubectl port-forward`ed pod.

Each user fires its requests one at a time, in order. Users run concurrently
with each other.

Usage:
    python bench_llm.py --concurrency 10 --requests-per-user 5
    kubectl port-forward pod/<name> 7860:7860 &
    python bench_llm.py --url http://127.0.0.1:7860/llm-demo --concurrency 20 --requests-per-user 3

Requires model_weights/qwen3_06b to be real weights, not LFS pointers
(`git lfs pull`), on whatever process is actually serving --url.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid

import httpx

API_NAME = "/on_start"  # apps/llm_demo.py: start_btn.click(fn=on_start, ...)
LOAD_API_NAME = "/load_model"  # apps/llm_demo.py: demo.load(fn=load_model, ...)

DUMMY_PROMPTS = [
    "The quick brown fox jumps over the lazy dog because",
    "In machine learning, overfitting occurs when",
    "The capital of France is",
    "Once upon a time, in a distant galaxy,",
    "def fibonacci(n):",
]


async def find_fn_index(client: httpx.AsyncClient, api_name: str) -> int:
    config = (await client.get("/config")).json()
    name = api_name.lstrip("/")
    deps = config["dependencies"]
    for dep in deps:
        if dep.get("api_name") == name:
            return dep["id"] if "id" in dep else deps.index(dep)
    raise RuntimeError(f"no endpoint named {api_name!r} in /config")


async def call_endpoint(
    client: httpx.AsyncClient, fn_index: int, data: list, timeout: float = 120.0
) -> tuple[bool, float, str | None]:
    """POST queue/join, drain queue/data over SSE, return (success, latency_ms, error)."""
    session_hash = uuid.uuid4().hex
    start = time.monotonic()
    try:
        resp = await client.post(
            "/gradio_api/queue/join",
            json={"data": data, "fn_index": fn_index, "session_hash": session_hash},
        )
        resp.raise_for_status()

        async with client.stream(
            "GET",
            "/gradio_api/queue/data",
            params={"session_hash": session_hash},
            timeout=timeout,
        ) as stream:
            buffer = b""
            async for chunk in stream.aiter_bytes():
                buffer += chunk
                while b"\n\n" in buffer:
                    message, buffer = buffer.split(b"\n\n", 1)
                    line = message.decode("utf-8").rstrip("\n")
                    if not line.startswith("data:"):
                        continue
                    msg = json.loads(line[5:])
                    kind = msg.get("msg")
                    if kind == "process_completed":
                        elapsed = (time.monotonic() - start) * 1000
                        success = bool(msg.get("success", True))
                        error = None if success else str(msg.get("output", {}).get("error"))
                        return success, elapsed, error
                    if kind == "close_stream":
                        return False, (time.monotonic() - start) * 1000, "stream closed without completion"
        return False, (time.monotonic() - start) * 1000, "stream ended without completion"
    except Exception as exc:
        return False, (time.monotonic() - start) * 1000, f"{type(exc).__name__}: {exc}"


async def warm_up(client: httpx.AsyncClient) -> None:
    """Fire /load_model once. demo.load() only runs on a real page visit, never
    on a direct /queue/join, so skipping this leaves tokenizer/model as None
    and every /on_start call crashes."""
    fn_index = await find_fn_index(client, LOAD_API_NAME)
    success, elapsed, error = await call_endpoint(client, fn_index, [])
    if not success:
        print(f"warmup failed: {error}", file=sys.stderr)
        sys.exit(1)
    print(f"warmup ok in {elapsed:.0f}ms")


async def run_user(
    client: httpx.AsyncClient,
    user_id: int,
    requests_per_user: int,
    fn_index: int,
    top_k: int,
    max_steps: int,
) -> list[tuple[bool, float, str | None]]:
    results = []
    for req_id in range(requests_per_user):
        prompt = DUMMY_PROMPTS[(user_id + req_id) % len(DUMMY_PROMPTS)]
        success, elapsed, error = await call_endpoint(
            client, fn_index, [prompt, top_k, max_steps]
        )
        print(
            f"user={user_id:>3} req={req_id:>3} "
            f"{'ok' if success else 'FAIL'} {elapsed:.0f}ms"
            + (f" ({error})" if error else "")
        )
        results.append((success, elapsed, error))
    return results


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7860/llm-demo", help="base URL of the mounted demo (main.py serves it under /llm-demo)")
    parser.add_argument("--concurrency", type=int, default=5, help="number of concurrent users")
    parser.add_argument("--requests-per-user", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10, help="keep low: each step is one forward pass")
    args = parser.parse_args()

    async with httpx.AsyncClient(base_url=args.url, timeout=120.0) as client:
        await warm_up(client)
        fn_index = await find_fn_index(client, API_NAME)

        start = time.monotonic()
        per_user = await asyncio.gather(
            *(
                run_user(client, user_id, args.requests_per_user, fn_index, args.top_k, args.max_steps)
                for user_id in range(args.concurrency)
            )
        )
        wall_s = time.monotonic() - start

    results = [r for user_results in per_user for r in user_results]
    latencies = sorted(elapsed for success, elapsed, _ in results if success)
    ok = sum(success for success, _, _ in results)

    print()
    print(f"total={len(results)} ok={ok} failed={len(results) - ok} wall={wall_s:.1f}s")
    if latencies:
        p50 = statistics.median(latencies)
        p90 = latencies[int(len(latencies) * 0.9) - 1]
        print(f"latency p50={p50:.0f}ms p90={p90:.0f}ms max={latencies[-1]:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())
