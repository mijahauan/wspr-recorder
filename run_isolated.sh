#!/bin/bash
# Run wspr-recorder on specific CPUs to avoid contention with radiod/hf-timestd
# Assumes 4+ core system.
# Adjust mask (-c) as needed:
# -c 3     -> CPU 3
# -c 2,3   -> CPUs 2 and 3
#
# radiod usually runs on all cores or is not pinned.
# hf-timestd might be heavy on one core.
# We pin this recorder to the last available core(s) to isolate it.

echo "Starting wspr-recorder on isolated CPUs (checking lscpu...)"
CPU_COUNT=$(nproc)

if [ "$CPU_COUNT" -gt 1 ]; then
  # Use the last core available
  AFFINITY="$((CPU_COUNT-1))"
  echo "Detected $CPU_COUNT cores. Isolating to CPU $AFFINITY."
else
  # Single core (cannot isolate)
  AFFINITY="0"
  echo "Single core detected. Cannot isolate."
fi

# Use taskset to pin CPU
# Use chrt to set FIFO priority (requires root/sudo usually, so we might skip or try)
# If user is not root, chrt usually fails. We'll stick to taskset.

taskset -c $AFFINITY python3 -m wspr_recorder -c config.toml
