import json, time, statistics, pathlib, requests

API = "http://localhost:8000/ask"
DATA = pathlib.Path("eval/testset.jsonl")

def ask(q: str):
    t0 = time.time()
    r = requests.post(API, json={"question": q}, timeout=60)
    dt = (time.time() - t0) * 1000
    r.raise_for_status()
    return r.json(), dt

def has_citation_text(ans: str):
    return "(" in ans and " p." in ans and ")" in ans

def main():
    latencies = []
    ok, total = 0, 0
    for line in DATA.read_text(encoding="utf-8").splitlines():
        total += 1
        j = json.loads(line)
        q = j["q"]
        expect_cit = j.get("expect_citation", True)
        res, dt = ask(q)
        latencies.append(dt)
        ans = res.get("answer","")
        # minimal groundedness check
        cited = has_citation_text(ans)
        passed = (cited == expect_cit)
        print(f"- Q: {q}\n  cited={cited}, expect={expect_cit}, latency_ms={dt:.0f}")
        if passed:
            ok += 1

    print(f"\nPassed {ok}/{total} | p50={statistics.median(latencies):.0f}ms p95={statistics.quantiles(latencies, n=20)[18]:.0f}ms")

if __name__ == "__main__":
    main()
