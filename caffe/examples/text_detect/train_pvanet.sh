#!/bin/bash
time=`date +"%m-%d-%H-%M"`
#export CAFFEROOT='/data1/liuxuebo/sensenet/example'
export CAFFEROOT='/mnt/lustre/liuxuebo/caffe_text_detect'
CAFFEBIN=$CAFFEROOT/build/tools/caffe
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CAFFEROOT/build/lib
export PYTHONPATH=$CAFFEROOT/python/src:$CAFFEROOT/python
srun  --gres=gpu:4 -n1 --ntasks-per-node=1  -p OCR $CAFFEBIN train --solver=./solver_pvanet.pt --weights=models/pva9.1_preAct_train_iter_1900000.caffemodel --gpu=0,1,2,3 2>&1| tee log/train_$time.log
#mpirun -n 1 $CAFFEBIN train --solver=./proto/solver.prototxt 2>&1 | tee train_ld.log

# ./evaluate.py 0.95 0.5
# ./utils/deteval.py det_res_0.95_0.50.txt ./dataset/1w/te_rois.txt res_0.95_0.50.txt
