#!/bin/bash
cd /home/dsi/michaer8/tralo-rebuild/lab/losslab
rm -f rows_*.json
for n in S1 S1_cap50 S1_c90 S1_c30 S3a S3b S4 S5a S5b_cap; do
  THREADS=1 bash run.sh $n 0 12 > log_${n}_0.txt 2>&1 &
  THREADS=1 bash run.sh $n 12 12 > log_${n}_12.txt 2>&1 &
done
for n in S2 S2_noise; do
  for s in 0 4 8 12 16 20; do THREADS=2 bash run.sh $n $s 4 > log_${n}_$s.txt 2>&1 & done
done
wait
