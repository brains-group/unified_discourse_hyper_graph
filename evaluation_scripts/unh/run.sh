#!/usr/bin/env bash
# UNH on one dataset, in the foreground: construct, then evaluate both views (full, hypergraph) one after the other.
# For both datasets with constructions in parallel and evaluations queued one at a time, use launch_unh.sh.
# Usage: bash evaluation_scripts/unh/run.sh hotpotqa|musique      (env: see pipeline.sh)
set -euo pipefail
DATASET="${1:?usage: run.sh hotpotqa|musique}"
HERE="$(cd "$(dirname "$0")" && pwd)"
bash "$HERE/pipeline.sh" construct "$DATASET"
bash "$HERE/pipeline.sh" evaluate "$DATASET"
