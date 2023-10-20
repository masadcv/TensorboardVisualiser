# TensorboardVisualiser

## Setup Instructions
1. Create virtualenv under: `/opt/virtualenvs/venv`
2. Create a folder `/opt/tf_tools` and place the files from this repo there
3. Update permissions for script as `chmod +x tensorboard.sh`
4. Create an alias in `~/.bash_rc` with contents `alias tb=/opt/tf_tools/tensorboard.sh`

## Instructions to run
1. Open terminal and cd to your folder that contains a single pb file to be visualised, e.g. `frozen_graph.pb` under `/models/mymodelgraph/`
2. Type: `tb` to run tensorboard visuliser
3. Open browser and go to: http://localhost:6006 to view the graph
4. Once done, close the terminal with ctrl+c to cleanup and close tensorboard session
