"""Check the set-up before the daily runs: which configured models the GitHub Models catalog offers and
at what rate-limit tier, that the MMLU test file downloads at the configured revision, and that one
call per model returns a parseable answer. Uses one request per model.

    GITHUB_TOKEN=... python probe.py --out results/probe.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from common import Client, load_config, load_questions, messages, parse_answer, resolve_models, sample_order


def call_with_headers(endpoint: str, token: str, model: str, msgs, cfg) -> dict:
    payload = {"model": model, "messages": msgs, "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"]}
    req = urllib.request.Request(endpoint.rstrip("/") + "/inference/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST", headers={
                                     "Accept": "application/vnd.github+json",
                                     "Authorization": f"Bearer {token}",
                                     "X-GitHub-Api-Version": "2022-11-28",
                                     "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode())
            headers = {k.lower(): v for k, v in resp.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
            text = body["choices"][0]["message"]["content"]
            return {"status": resp.status, "model_reported": body.get("model"), "text": text,
                    "answer": parse_answer(text), "finish_reason": body["choices"][0].get("finish_reason"),
                    "usage": body.get("usage"), "rate_headers": headers}
    except urllib.error.HTTPError as e:
        headers = {k.lower(): v for k, v in e.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
        return {"status": e.code, "error": e.read().decode(errors="replace")[:1000], "rate_headers": headers}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results/probe.json"))
    ap.add_argument("--no-calls", action="store_true", help="catalog and questions only")
    ap.add_argument("--endpoint", help="override the API endpoint (tests)")
    ap.add_argument("--questions-file", type=Path, help="synthetic questions (tests); default MMLU")
    args = ap.parse_args(argv)
    cfg = load_config()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")
    report = {"probed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    endpoint = args.endpoint or cfg["endpoint"]
    client = Client(endpoint, token or "")
    try:
        catalog = client.catalog()
        report["catalog_size"] = len(catalog)
        report["catalog"] = sorted(({"id": m.get("id"), "tier": m.get("rate_limit_tier"),
                                     "publisher": m.get("publisher")} for m in catalog), key=lambda m: m["id"] or "")
        specs = resolve_models(cfg, catalog)
        report["resolved"] = [{k: s[k] for k in ("key", "family", "id", "tier", "n_questions", "rpm", "per_day")}
                              for s in specs]
    except Exception as e:  # noqa: BLE001 - the probe reports whatever went wrong
        report["catalog_error"] = repr(e)
        specs = []
    try:
        rows, source = load_questions(cfg, args.questions_file)
        order = sample_order(rows, cfg)
        report["questions"] = {**source, "subjects": len({r.subject for r in rows}),
                               "sampled": len(order), "sampled_subjects": len({q.subject for q in order}),
                               "keyed_answers": dict(Counter("ABCD"[q.answer] for q in order))}
    except Exception as e:  # noqa: BLE001
        report["questions_error"] = repr(e)
        order = []
    if not args.no_calls and token and order:
        q = order[0]
        report["calls"] = {}
        for s in specs:
            if s["id"]:
                report["calls"][s["key"]] = call_with_headers(endpoint, token, s["id"],
                                                              messages("coder", q), cfg)
        report["calls_question"] = {"qid": q.qid, "subject": q.subject, "keyed": "ABCD"[q.answer]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "catalog"}, indent=2, ensure_ascii=False))
    ok = "catalog_error" not in report and "questions_error" not in report
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
