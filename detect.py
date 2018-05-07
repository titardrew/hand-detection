"""This script performs hand-detection."""
import numpy as np
import tensorflow as tf
import visualization_utils as vis_util
from PIL import Image
import os
from tqdm import tqdm
from itertools import combinations

flags = tf.app.flags
flags.DEFINE_string('image_path', 'raw_data/test', 'Path to image, or'
                    ' directory with images (.jpg)')
flags.DEFINE_integer('limit', 0, 'If image_path is a path to directory'
                 ' this limits number of images to the particular number.')
flags.DEFINE_string('frozen_path',
                    'fine_tuned_model/frcnn_inc_v2_aug5/optimized_inference_graph.pb',
                    'Path to frozen graph of the model')
flags.DEFINE_string('output_dir', 'predicted_images/frcnn_inc_v2_aug5',
                    'This path will be used to store output images in.')

FLAGS = flags.FLAGS


class HandDetector(object):
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FLAGS.frozen_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def predict(self, img):
        # Bounding Box Detection.
        with self.detection_graph.as_default():
            # Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(img, axis=0)  
            (boxes, scores, classes, num) = self.sess.run(
                [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                feed_dict={self.image_tensor: img_expanded})
        return boxes, scores, classes, num

    def predict_and_save_image(self, image_path):
        """Perform inference and draw bboxes."""
        name = os.path.split(image_path)[-1]
        img = Image.open(image_path)
        boxes, scores, classes, num = self.predict(img)
        image_np = load_image_into_numpy_array(img)

        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes[0],
            classes[0].astype(np.uint8),
            scores[0],
            {1: {'id': 1, 'name': 'left'},
            2: {'id': 2, 'name': 'right'}},
            use_normalized_coordinates=True,
            line_thickness=1)

        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)

        Image.fromarray(image_np).save(
            os.path.join(FLAGS.output_dir, name))


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def main(_):
    detector = HandDetector()
    path = FLAGS.image_path

    if os.path.isdir(path):
        dir_ = path
        files = [f for f in os.listdir(dir_)
            if os.path.isfile(os.path.join(dir_, f))
                 and f[-4:] == '.jpg']

        if FLAGS.limit > 0:
            files = files[:FLAGS.limit]

        for img in tqdm(files):
            detector.predict_and_save_image(os.path.join(dir_, img))
    elif os.path.isfile(path):
        if path[-4:] == '.jpg':
            detector.predict_and_save_image(path)
        else:
            print("Wrong image type! It should be .jpg.")
    else:
        print("Wrong image_path = %s! Provide a path to image or directory."
              % path)


if __name__=="__main__":
    tf.app.run()
