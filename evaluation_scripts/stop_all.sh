#!/usr/bin/env bash
# Stop every job started by launch_all.sh / launch_quality.sh / launch_gmedical.sh / launch_contracts.sh (each runs in its own process group; the group id is in run_logs/*.pid).
# Finished work is kept: LLM calls are cached, so rerunning launch_all.sh resumes cheaply.
EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
shopt -s nullglob
for f in "$EVAL_ROOT"/run_logs/*_hotpotqa.pid "$EVAL_ROOT"/run_logs/*_musique.pid "$EVAL_ROOT"/run_logs/*_quality*.pid "$EVAL_ROOT"/run_logs/*_gmedical*.pid "$EVAL_ROOT"/run_logs/unh_pipeline*.pid "$EVAL_ROOT"/run_logs/*_contracts_*.pid; do
  pid="$(cat "$f")"
  if kill -0 "$pid" 2>/dev/null; then kill -- "-$pid" && echo "stopped $(basename "$f" .pid)"; else echo "not running: $(basename "$f" .pid)"; fi
done
