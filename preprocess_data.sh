#! /bin/bash
DATASET="kitti360"
SEQ_ID="4950"

python -m data.preprocess.generate_rangeview --dataset $DATASET --sequence_id $SEQ_ID

python -m data.preprocess.kitti360_to_nerf --sequence_id $SEQ_ID

python -m data.preprocess.cal_seq_config --dataset $DATASET --sequence_id $SEQ_ID