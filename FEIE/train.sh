#/bin/bash

DATA_DIR=${HOME}/DISK_2T/datasets/FEID/dataset

for e in Anger Disgust Fear Happiness Sadness Surprise; do
    base=FEIE_$e
    echo $base

python core/train.py \
    --lr=0.00001 \
    --gpu=1 \
    --steps=45 \
    --epochs=30 \
    --batch_size=16 \
    --log=log/${base}.txt \
    --model_dir=model/${base} \
    --res=res/${base}.txt \
    --print_freq=1 \
    --ckpt_save_freq=15 \
    --test_freq=1 \
    --arch=vgg16 \
    --train_dataset=FEIDatasetTrain \
    --test_dataset=FEIDatasetTest \
    --dataset_dir=${DATA_DIR} \
    --fold_index=0 \
    --pretrain=model/vgg16_1cls_state_dict.pth \
    --emotion=$e \
    --R=1 \
    --lstm_output=weight_multi;
done

#--only_test;
