#!/bin/bash
cd /home/dsi/michaer8/tralo-rebuild/lab/losslab
mkdir -p v1; mv rows_*.json v1/ 2>/dev/null
for n in S1 S3b S5a S5b_cap S1_cap50; do
  THREADS=1 bash run.sh $n 0 12 > log_${n}_0.txt 2>&1 &
  THREADS=1 bash run.sh $n 12 12 > log_${n}_12.txt 2>&1 &
done
for s in 0 4 8 12 16 20; do THREADS=2 bash run.sh S2_noise $s 4 > log_S2_noise_$s.txt 2>&1 & done
wait
