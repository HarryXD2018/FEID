#/bin/bash
MODEL=feid_fer
DATA_DIR=${HOME}/DISK_2T/datasets/FEID/dataset

python core/train.py \
    --lr=0.01 \
    --gpu=1 \
    --steps=30 \
    --epochs=50 \
    --batch_size=1 \
    --num_classes=7 \
    --log=log/${MODEL}.txt \
    --model_dir=model/${MODEL} \
    --ckpt_save_freq=10 \
    --arch=vgg16 \
    --train_dataset=FEIDClsTrain \
    --test_dataset=FEIDClsTest \
    --base_dir=${DATA_DIR} \
    --score_save_dir=predict_score \
    --pretrain=model/vgg16_7cls_state_dict.pth \
