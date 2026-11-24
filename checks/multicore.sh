#!/bin/bash
# multicore.sh -- run examples/multicore at several thread counts (the digests must agree across them), then time
# each parallel task on one thread for checks/scaling_sim.py. Run after `cargo build --release --examples`.
# Eliot R. Weinstein / eliotw@uchicago.edu / erweinstein.com
set -e
echo "# machine: $(nproc) cores; $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'); $(free -g | awk '/Mem:/ {print $2}') GB"
echo "# rustc: $(rustc --version)"
for t in ${THREADS:-1 2 3 4 6 8}; do
  RAYON_NUM_THREADS=$t ./target/release/examples/multicore
done
RAYON_NUM_THREADS=1 ./target/release/examples/multicore --tasks
