thread_num=$1
for ((i=0;i<$thread_num;i++))
do
    srun  --gres=gpu:1 -n1 --ntasks-per-node=1  -p OCR python unitbox_test_old.py $thread_num $i &
done
