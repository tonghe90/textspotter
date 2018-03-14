#!/bin/bash
time=`date +"%m-%d-%H-%M"`
#export CAFFEROOT='/data1/liuxuebo/sensenet/example'
export CAFFEROOT='/mnt/lustre/chendagui/text_detect'
CAFFEBIN=$CAFFEROOT/build/tools/caffe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CAFFEROOT/build/lib
export PYTHONPATH=$CAFFEROOT/python
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_CMA=0 srun  --mpi=pmi2 --gres=gpu:4 -n 1 -p Single --job-name=text_detect $CAFFEBIN train --solver=./CNN_solver.prototxt --weights=./snapshot/pre_CNN_iter_10000.caffemodel 2>&1| tee ./logs/train_$time.log &
#mpirun -n 1 $CAFFEBIN train --solver=./proto/solver.prototxt 2>&1 | tee train_ld.log

# ./evaluate.py 0.95 0.5
# ./utils/deteval.py det_res_0.95_0.50.txt ./dataset/1w/te_rois.txt res_0.95_0.50.txt
