# ! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --outdir=./training-runs                                    \
    --cfg=ffhq                                                  \
    --data=/path/to/dataset/                                    \
    --gpus=4                                                    \
    --batch-gpu=2                                               \
    --mbstd-group=2                                             \
    --batch=16                                                  \
    --gamma=1                                                   \
    --gen_pose_cond=True                                        \
    --g_num_deformable_res=4                                    \
    --deform_type=template+mapping                              \
    --resume=../data/ffhq512-64-rolled.pkl