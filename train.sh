#!/bin/bash
# Script performs training of the model.

export PYTHONPATH=$PYTHONPATH:`pwd`/models/research/:`pwd`/models/research/slim
python3 train.py --logtostderr --train_dir=./train/frcnn_r50_coco --pipeline_config_path=./faster_rcnn_resnet50_coco.config
