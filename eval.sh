#!/bin/bash
# Script performs evaluation of the model. 

export PYTHONPATH=$PYTHONPATH:`pwd`/models/research/:`pwd`/models/research/slim
python3 eval.py --logtostderr --checkpoint_dir=train/frcnn_gpu --eval_dir=./eval --pipeline_config_path=./faster_rcnn_resnet50_coco.config
