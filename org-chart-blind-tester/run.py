"""Ask every configured model the sampled questions in the four roles, resuming where earlier runs
stopped. Meant to run once a day: the free GitHub Models quota is a number of requests per model per
day, so each run works until a model's quota is used up or its questions are done, and appends every
response to runs/<model key>.jsonl as soon as it arrives.

    GITHUB_TOKEN=... python run.py --runs-dir runs

The token needs the models:read permission (in Actions: `permissions: models: read`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

from common import (Client, QuotaExhausted, Rejected, ROLES, in_check_set, load_config, load_questions,
                    messages, parse_answer, read_runs, resolve_models, sample_order)


def needs_followup(coder: dict, blind: dict, qid: int, cfg: dict) -> bool:
    return in_check_set(qid, cfg) or coder.get("answer") is None or blind.get("answer") != coder.get("answer")


def pending_steps(order, done: dict, cfg: dict):
    """Yield (position, question, role) still to ask, in order. The anchored and warned roles of a
    question are decided only once its coder and blind answers exist."""
    for pos, q in enumerate(order):
        c, b = done.get((q.qid, "coder")), done.get((q.qid, "blind"))
        if c is None:
            yield pos, q, "coder"
            c = done.get((q.qid, "coder"))
            if c is None:          # the caller stopped before recording it
                return
        if b is None:
            yield pos, q, "blind"
            b = done.get((q.qid, "blind"))
            if b is None:
                return
        if needs_followup(c, b, q.qid, cfg):
            for role in ("anchored", "warned"):
                if (q.qid, role) not in done:
                    yield pos, q, role
                    if (q.qid, role) not in done:
                        return


def question_complete(q, done: dict, cfg: dict) -> bool:
    c, b = done.get((q.qid, "coder")), done.get((q.qid, "blind"))
    if c is None or b is None:
        return False
    if needs_followup(c, b, q.qid, cfg):
        return (q.qid, "anchored") in done and (q.qid, "warned") in done
    return True


def run_model(spec: dict, questions, cfg: dict, client: Client, runs_dir: Path, max_calls: int,
              max_seconds: float, log=print) -> dict:
    path = runs_dir / f"{spec['key']}.jsonl"
    done = {(r["qid"], r["role"]): r for r in read_runs(path)}
    order = questions[: spec["n_questions"]]
    interval = 60.0 / spec["rpm"] * 1.05
    calls, stopped, last, t0 = 0, "complete", 0.0, time.monotonic()
    with path.open("a") as fh:
        for pos, q, role in pending_steps(order, done, cfg):
            if calls >= max_calls:
                stopped = "max_calls"
                break
            if time.monotonic() - t0 > max_seconds:
                stopped = "time"
                break
            wait = interval - (time.monotonic() - last)
            if wait > 0:
                client.sleep(wait)
            coder_text = done[(q.qid, "coder")].get("text") if role in ("anchored", "warned") else None
            rec = {"experiment": cfg["experiment"], "prompt_version": cfg["prompt_version"],
                   "key": spec["key"], "model_id": spec["id"], "role": role, "qid": q.qid,
                   "position": pos, "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"]}
            try:
                last = time.monotonic()
                resp = client.chat(spec["id"], messages(role, q, coder_text or ""),
                                   cfg["temperature"], cfg["max_tokens"])
                choice = resp["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
                rec.update(text=text, answer=parse_answer(text), finish_reason=choice.get("finish_reason"),
                           usage=resp.get("usage"), model_reported=resp.get("model"), error=None)
            except QuotaExhausted as e:
                stopped = "quota"
                log(f"[{spec['key']}] daily quota reached after {calls} calls: {e}")
                break
            except Rejected as e:
                rec.update(text=None, answer=None, finish_reason=None, usage=None, model_reported=None,
                           error=str(e)[:300])
            except RuntimeError as e:
                stopped = "error"
                log(f"[{spec['key']}] stopping on error: {e}")
                break
            calls += 1
            rec["created"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[(q.qid, role)] = rec
    complete = sum(question_complete(q, done, cfg) for q in order)
    if stopped == "complete" and complete < len(order):
        stopped = "incomplete"
    log(f"[{spec['key']}] {spec['id']}: {calls} calls this run, {complete}/{len(order)} questions complete, "
        f"stopped: {stopped}")
    return {"key": spec["key"], "id": spec["id"], "tier": spec["tier"], "calls": calls,
            "questions_complete": complete, "questions": len(order), "stopped": stopped,
            "complete": complete == len(order)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--models", nargs="*", help="model keys from config.json; default all")
    ap.add_argument("--questions-file", type=Path, help="synthetic questions (tests); default MMLU")
    ap.add_argument("--endpoint", help="override the API endpoint (tests)")
    ap.add_argument("--max-calls", type=int, help="per model in this run; default the tier's daily quota")
    ap.add_argument("--max-minutes", type=float, default=150.0)
    ap.add_argument("--status", type=Path, help="write a JSON status summary here")
    args = ap.parse_args(argv)

    cfg = load_config()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")
    if not token:
        print("GITHUB_TOKEN is not set", file=sys.stderr)
        return 2
    client = Client(args.endpoint or cfg["endpoint"], token)
    rows, source = load_questions(cfg, args.questions_file)
    order = sample_order(rows, cfg)
    specs = resolve_models(cfg, client.catalog())
    if args.models:
        specs = [s for s in specs if s["key"] in args.models]
    missing = [s["key"] for s in specs if s["id"] is None]
    for key in missing:
        print(f"[{key}] none of its candidate ids is in the catalog; skipped", file=sys.stderr)
    specs = [s for s in specs if s["id"] is not None]
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    results, lock = [], threading.Lock()

    def worker(spec):
        res = run_model(spec, order, cfg, client, args.runs_dir,
                        args.max_calls if args.max_calls is not None else spec["per_day"],
                        args.max_minutes * 60)
        with lock:
            results.append(res)

    threads = [threading.Thread(target=worker, args=(s,)) for s in specs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    status = {"finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "questions_source": source, "missing_models": missing,
              "models": sorted(results, key=lambda r: r["key"]),
              "all_complete": bool(results) and all(r["complete"] for r in results) and not missing}
    text = json.dumps(status, indent=2)
    print(text)
    if args.status:
        args.status.write_text(text + "\n")
    return 1 if any(r["stopped"] == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
