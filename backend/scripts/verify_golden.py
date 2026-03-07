"""Verify golden dataset against RunPod RAG API using stdlib only."""
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

API_BASE = "https://7rzubyo9fsfmco-8000.proxy.runpod.net"
DATASET_PATH = Path(__file__).parent.parent / "tests" / "golden_dataset_evaluation.json"


def api_post(path: str, data: dict, token: str = "") -> Optional[dict]:
    """POST request using urllib."""
    url = f"{API_BASE}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "Mozilla/5.0 (golden-verify/1.0)")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  HTTP {e.code}: {body[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return None


def main():
    # Flush stdout for real-time output
    print("=== Golden Dataset Verification ===", flush=True)

    # Login
    print("Logging in...", flush=True)
    resp = api_post("/api/auth/login", {"username": "admin", "password": "admin123"})
    if not resp or "access_token" not in resp:
        print("Login failed!", flush=True)
        sys.exit(1)
    token = resp["access_token"]
    print("Login OK.", flush=True)

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    questions = dataset["questions"]
    total = len(questions)
    print(f"Questions: {total}\n", flush=True)

    header = f"{'#':>3} {'Category':<10} {'Stat':<5} {'Time':>6} Question"
    sep = f"{'---':>3} {'----------':<10} {'-----':<5} {'------':>6} {'----------------------------------'}"
    print(header, flush=True)
    print(sep, flush=True)

    ok = miss = err = 0
    times = []

    for i, q in enumerate(questions):
        num = i + 1
        start = time.time()

        resp = api_post("/api/chat", {
            "message": q["question"],
            "temperature": 0.1,
        }, token=token)

        elapsed = time.time() - start
        times.append(elapsed)

        if resp:
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            if len(answer) > 10:
                status = "OK"
                ok += 1
            elif q["category"] == "negative" and len(answer) > 0:
                status = "OK"
                ok += 1
            else:
                status = "MISS"
                miss += 1
        else:
            status = "ERR"
            err += 1

        short_q = q["question"][:32]
        print(f"{num:3d} {q['category']:<10} {status:<5} {elapsed:5.1f}s {short_q}", flush=True)

        time.sleep(0.3)

    # Summary
    avg_time = sum(times) / max(len(times), 1)
    print(f"\n{'='*60}", flush=True)
    print(f"Results: {ok}/{total} OK ({ok/total*100:.1f}%)", flush=True)
    print(f"  OK:    {ok}", flush=True)
    print(f"  MISS:  {miss}", flush=True)
    print(f"  ERROR: {err}", flush=True)
    print(f"  Avg time: {avg_time:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)

    # Save results
    results_path = DATASET_PATH.parent / "golden_verification_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": total,
            "ok": ok,
            "miss": miss,
            "error": err,
            "success_rate": f"{ok/total*100:.1f}%",
            "avg_response_time": f"{avg_time:.1f}s",
        }, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
