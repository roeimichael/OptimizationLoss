#!/bin/bash
cd /home/dsi/michaer8/tralo-rebuild/lab/losslab
mkdir -p v3; mv rows_S1_*.json rows_S3a_*.json rows_S3b_*.json v3/ 2>/dev/null
for n in S1 S3a S3b S5a; do
  mkdir -p v3; mv rows_${n}_*.json v3/ 2>/dev/null
  THREADS=1 bash run.sh $n 0 12 > log_${n}_0.txt 2>&1 &
  THREADS=1 bash run.sh $n 12 12 > log_${n}_12.txt 2>&1 &
done
wait
