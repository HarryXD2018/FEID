#/bin/bash

DATA_DIR=${HOME}/DISK_2T/datasets/FEID/dataset

for e in Anger Disgust Fear Happiness Sadness Surprise; do
    base=FEIE_$e
    echo $base
    model_dir=model/$base

python core/train.py \
    --gpu=1 \
    --log=log/debug.txt \
    --res=res/debug.txt \
    --arch=vgg16 \
    --train_dataset=FEIDatasetTrain \
    --test_dataset=FEIDatasetTest \
    --dataset_dir=${DATA_DIR} \
    --emotion=$e \
    --resume=$model_dir/ckpt_fold0_best.pth \
    --only_test \
    --lstm_output=weight_multi \
    --fold_index=0 \
    --R=1 ;
done
