#!/bin/bash
# --model_name cls_b64e500s200lr1e-3_r020tr64-10_te32_bn1024 \
python3 /home/sumanth/PointSetVoting/utils/main.py \
--dataset_path /home/sumanth/PointSetVoting/data_root/evitado_data \
--task classification \
--num_pts 2048 \
--num_pts_observed 512 \
--lr 0.001 \
--step_size 300 \
--max_epoch 1000 \
--bsize 7 \
--radius 0.2 \
--bottleneck 512 \
--num_vote_train 64 \
--num_contrib_vote_train 10 \
--num_vote_test 32 \
--is_vote \
--is_simuOcc \
--norm scale \
--split_file split.json \

