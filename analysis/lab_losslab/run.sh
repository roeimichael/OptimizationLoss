#!/bin/bash
# usage: run.sh NAME START NSEEDS
cd /home/dsi/michaer8/tralo-rebuild/lab/losslab
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=${THREADS:-1} MKL_NUM_THREADS=${THREADS:-1}
export PYTHONPATH=/home/dsi/michaer8/tralo-rebuild/releases/c5634488788b83e3a80820127be16c4858d4772d:.
nice -n 19 timeout 590 /home/dsi/michaer8/anaconda3/envs/optloss/bin/python lab.py configs.json $1 --start $2 --seeds $3 --out rows_$1_$2.json
