import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import utils
import argparse

# load graph and create tensorboard graph at log_dir
def load_and_create_tensorboard_graph(pb_path, log_dir):
    graph_def = utils.load_frozen_graph(pb_path)
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    writer = tf.summary.FileWriter(log_dir, graph)
    writer.close()

def get_args():
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('--pb_path', required=True, help='Path to frozen graph')
    argsparser.add_argument('--log_dir', required=True, help='Path to log directory')
    return argsparser.parse_args()

if __name__ == '__main__':
    args = get_args()
    load_and_create_tensorboard_graph(args.pb_path, args.log_dir)