
import glob
import os
import json

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# load frozen graph
def load_frozen_graph(pb_path):
    with tf.io.gfile.GFile(pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def get_image_paths_from_folder(folder):
    jpg_images = glob.glob(os.path.join(folder, '*.jpg'))
    jpeg_images = glob.glob(os.path.join(folder, '*.jpeg'))
    png_images = glob.glob(os.path.join(folder, '*.png'))
    image_paths = jpg_images + jpeg_images + png_images
    return image_paths

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data
