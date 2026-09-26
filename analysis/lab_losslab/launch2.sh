#!/bin/bash
cd /home/dsi/michaer8/tralo-rebuild/lab/losslab
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/dsi/michaer8/tralo-rebuild/releases/c5634488788b83e3a80820127be16c4858d4772d:.
PY=/home/dsi/michaer8/anaconda3/envs/optloss/bin/python
rm -f rows_S4_*.json
for n in S4 S4_noshift; do
  THREADS=1 bash run.sh $n 0 12 > log_${n}_0.txt 2>&1 &
  THREADS=1 bash run.sh $n 12 12 > log_${n}_12.txt 2>&1 &
done
for n in S1 S3a S5a S5b_cap; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 timeout 590 $PY mech.py $n 24 > mech_$n.txt 2>&1 &
done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 timeout 590 $PY mech.py S2 10 > mech_S2.txt 2>&1 &
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 timeout 590 $PY edge.py S1 > edge_S1.txt 2>&1 &
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 timeout 590 $PY edge.py S2 > edge_S2.txt 2>&1 &
wait
