"""Ask a model the sampled questions in the four roles through a local Ollama server, resuming where an
earlier run stopped. The questions can be split into shards (position modulo the number of shards) so
that several jobs share a model's questions; each shard appends every response to
runs/<key>.shard<i>of<k>.jsonl as soon as it arrives.

    python run.py --models llama3.1-8b --shard 0 --shards 4 --runs-dir runs
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from common import (Ollama, in_check_set, load_config, load_questions, messages, model_specs, parse_answer_how,
                    read_jsonl, sample_order, shard_path)


def needs_followup(coder: dict, blind: dict, qid: int, cfg: dict) -> bool:
    return in_check_set(qid, cfg) or coder.get("answer") is None or blind.get("answer") != coder.get("answer")


def pending_steps(order, done: dict, cfg: dict):
    """Yield (position, question, role) still to ask, in order. The anchored and warned roles of a
    question are decided only once its coder and blind answers exist; the caller records each answer
    in `done` before asking for the next step."""
    for pos, q in order:
        if (q.qid, "coder") not in done:
            yield pos, q, "coder"
            if (q.qid, "coder") not in done:
                return
        if (q.qid, "blind") not in done:
            yield pos, q, "blind"
            if (q.qid, "blind") not in done:
                return
        if needs_followup(done[(q.qid, "coder")], done[(q.qid, "blind")], q.qid, cfg):
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


def run_shard(spec: dict, order, shard: int, shards: int, cfg: dict, client: Ollama, runs_dir: Path,
              max_calls: int | None, max_seconds: float, log=print) -> dict:
    path = shard_path(runs_dir, spec["key"], shard, shards)
    done = {(r["qid"], r["role"]): r for r in read_jsonl(path)}
    mine = [(pos, q) for pos, q in enumerate(order) if pos % shards == shard]
    calls, stopped, t0 = 0, "complete", time.monotonic()
    with path.open("a") as fh:
        for pos, q, role in pending_steps(mine, done, cfg):
            if max_calls is not None and calls >= max_calls:
                stopped = "max_calls"
                break
            if time.monotonic() - t0 > max_seconds:
                stopped = "time"
                break
            coder_text = done[(q.qid, "coder")].get("text") if role in ("anchored", "warned") else None
            rec = {"experiment": cfg["experiment"], "prompt_version": cfg["prompt_version"],
                   "key": spec["key"], "model": spec["model"], "role": role, "qid": q.qid, "position": pos,
                   "temperature": cfg["temperature"], "seed": cfg["seed"], "max_tokens": cfg["max_tokens"]}
            try:
                out = client.chat(spec, messages(role, q, coder_text or ""), cfg)
            except RuntimeError as e:
                stopped = "error"
                log(f"[{spec['key']} shard {shard}] stopping on error: {e}")
                break
            answer, how = parse_answer_how(out["text"])
            rec.update(out, answer=answer, parsed_by=how,
                       created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[(q.qid, role)] = rec
            calls += 1
            if calls % 25 == 0:
                rate = (time.monotonic() - t0) / calls
                log(f"[{spec['key']} shard {shard}] {calls} calls, {rate:.1f} s per call")
    complete = sum(question_complete(q, done, cfg) for _, q in mine)
    if stopped == "complete" and complete < len(mine):
        stopped = "incomplete"
    log(f"[{spec['key']} shard {shard}/{shards}] {calls} calls this run, {complete}/{len(mine)} questions "
        f"complete, stopped: {stopped}, {time.monotonic() - t0:.0f} s")
    return {"key": spec["key"], "model": spec["model"], "shard": shard, "shards": shards, "calls": calls,
            "questions_complete": complete, "questions": len(mine), "stopped": stopped,
            "complete": complete == len(mine), "seconds": round(time.monotonic() - t0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument("--models", nargs="*", help="model keys from config.json; default all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, help="default: the model's own setting in config.json")
    ap.add_argument("--questions-file", type=Path, help="synthetic questions (tests); default MMLU")
    ap.add_argument("--base-url", default="http://localhost:11434")
    ap.add_argument("--max-calls", type=int, help="stop after this many calls (per model)")
    ap.add_argument("--max-minutes", type=float, default=320.0)
    ap.add_argument("--status", type=Path, help="write a JSON status summary here")
    ap.add_argument("--n-questions", type=int, help="ask only the first N of the sampled order (smoke runs)")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.n_questions:
        cfg["n_questions"] = min(args.n_questions, cfg["n_questions"])
    client = Ollama(args.base_url)
    rows, source = load_questions(cfg, args.questions_file)
    order = sample_order(rows, cfg)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in model_specs(cfg, args.models):
        shards = args.shards or spec["shards"]
        if not 0 <= args.shard < shards:
            print(f"[{spec['key']}] shard {args.shard} out of range for {shards} shards", file=sys.stderr)
            return 2
        prov = client.provenance(spec)
        prov_path = shard_path(args.runs_dir, spec["key"], args.shard, shards).with_suffix(".provenance.json")
        prov_path.write_text(json.dumps(prov, indent=2) + "\n")
        results.append(run_shard(spec, order, args.shard, shards, cfg, client, args.runs_dir, args.max_calls,
                                 args.max_minutes * 60))
    status = {"finished_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "questions_source": source, "shards": results,
              "all_complete": bool(results) and all(r["complete"] for r in results)}
    text = json.dumps(status, indent=2)
    print(text)
    if args.status:
        args.status.write_text(text + "\n")
    return 1 if any(r["stopped"] == "error" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
