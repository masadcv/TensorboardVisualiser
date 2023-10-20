#!/bin/bash

# create tensorboard output using create_tensorboard_graph
# and then run tensorboard

# Usage: tensorboard.sh <port>
port=$1
# if port empty then initial to 6006 value
if [ -z "$port" ]
then
    port=6006
fi

term() {
    # echo ctrl c pressed!
    # capture kill signal and remove tensorboard directory
    rm -r $tb_directory
}
trap term SIGINT

# get current directory
current_directory=$(pwd)
tb_directory=${current_directory}/tensorboard

echo "current directory: $current_directory"
echo "tensorboard directory: $tb_directory"
echo "${current_directory}/*.pb"

# create tensorboard output
/opt/virtualenvs/venv/bin/python /opt/tf_tools/create_tensorboard_graph.py --pb_path ${current_directory}/*.pb --log_dir $tb_directory

# run tensorboard
/opt/virtualenvs/venv/bin/tensorboard --logdir=$tb_directory --port=$port

