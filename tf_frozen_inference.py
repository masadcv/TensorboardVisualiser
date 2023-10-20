
import argparse
import os

import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import utils

tf.disable_v2_behavior()

# get input and output tensors from graph
def get_input_output_tensors(graph, input_nodes, output_nodes):
    input_tensors = []
    output_tensors = []
    for input_node in input_nodes:
        input_tensors.append(graph.get_tensor_by_name(input_node))
    for output_node in output_nodes:
        output_tensors.append(graph.get_tensor_by_name(output_node))
    return input_tensors, output_tensors


# run inference
def run_inference(graph, input_tensors, output_tensors, input_data):
    with tf.Session(graph=graph) as sess:
        output_data = sess.run(
            output_tensors, feed_dict=dict(zip(input_tensors, input_data))
        )

    output_data = {
        output_tensors[i].name: output_data[i].tolist()
        for i in range(len(output_tensors))
    }
    return output_data


def load_image_data(image_path, image_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size[1], image_size[0]))
    if 1 == 0:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imshow("image", image)
    # cv2.waitKey(20)
    # import matplotlib.pyplot as plt
    # plt.imshow(image); plt.show()

    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


# load graph and run inference
def load_and_run_inference(pb_path, input_nodes, output_nodes, input_image_paths):
    assert len(input_image_paths) > 0, "No input images found"

    # load graph
    graph_def = utils.load_frozen_graph(pb_path)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    # get input and output tensors from graph
    input_tensors, output_tensors = get_input_output_tensors(
        graph, input_nodes, output_nodes
    )

    # load input data
    input_shape = utils.shape(input_tensors[0])
    input_data = [
        load_image_data(x, (input_shape[1], input_shape[2])) for x in input_image_paths
    ]

    # run inference
    output_data = {}
    for idname, id in zip(input_image_paths, input_data):
        idname = os.path.splitext(os.path.basename(idname))[0]
        output_data[idname] = run_inference(graph, input_tensors, output_tensors, [id])
    return output_data


# get args
def get_args():
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--pb_path", required=True, help="Path to frozen graph")
    argsparser.add_argument("--input_nodes", required=True, help="Input nodes")
    argsparser.add_argument("--output_nodes", required=True, help="Output nodes")
    argsparser.add_argument("--input_data", required=True, help="Input data")
    argsparser.add_argument(
        "--output_data", default="infer_out", required=False, help="Output data folder"
    )
    return argsparser.parse_args()


if __name__ == "__main__":
    args = get_args()
    input_nodes = args.input_nodes.split(",")
    output_nodes = args.output_nodes.split(",")
    input_data = utils.get_image_paths_from_folder(args.input_data)
    output_data = load_and_run_inference(
        args.pb_path, input_nodes, output_nodes, input_data
    )
    model_file = os.path.splitext(os.path.basename(args.pb_path))[0]
    os.makedirs(args.output_data, exist_ok=True)
    utils.save_json(output_data, os.path.join(args.output_data, f"{model_file}.json"))
