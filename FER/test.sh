#/bin/bash

MODEL=feid_fer
DATA_DIR=${HOME}/DISK_2T/datasets/FEID/dataset

python core/train.py \
    --gpu=1 \
    --num_classes=7 \
    --arch=vgg16 \
    --train_dataset=FEIDClsTest \
    --test_dataset=FEIDClsTest \
    --base_dir=${DATA_DIR} \
    --score_save_dir=predict_score \
    --resume=model/${MODEL}/ckpt_best.pth \
    --log=log/test.txt \
    --only_test
