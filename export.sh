#!/bin/bash
# Script exports trained model as inference_graph.

python3 export_inference_graph.py \
       --input_type image_tensor \
       --pipeline_config_path frcnn_inc_v2_aug.config \
       --trained_checkpoint_prefix train/frcnn_gpu_aug/model.ckpt-20050 \
       --output_directory fine_tuned_model/frcnn_inc_v2_aug5
