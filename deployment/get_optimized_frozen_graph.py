import os

import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.core.framework import graph_pb2

def main(saves_dir='../fine_tuned_model'):
    # load previously frozen graph
    frozen_save_path = os.path.join(saves_dir, 'frcnn_inc_v2_aug4/frozen_inference_graph.pb')
    input_graph_def = graph_pb2.GraphDef()
    with tf.gfile.Open(frozen_save_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    # optimize graph
    optimized_constant_graph = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ['image_tensor'],
        ['detection_boxes', 'detection_scores',
         'detection_classes', 'num_detections'],
        tf.uint8.as_datatype_enum
    )

    # save optimized graph
    optimized_graph_path = os.path.join(saves_dir, 'frcnn_inc_v2_aug4/optimized_graph.pb')
    with tf.gfile.GFile(optimized_graph_path, "wb") as f:
        f.write(optimized_constant_graph.SerializeToString())


if __name__ == '__main__':
    main()
