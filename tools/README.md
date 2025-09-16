<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

[日本語版 README はこちら](https://github.com/MapIV/LION/tree/main/tools/README-ja.md)

# LION

## Installation

### Conda Installation

First, create a conda environment by executing the command below.

```bash
conda create -n LION python==3.9
conda activate LION
```

Next, install PyTorch by executing the command below.

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then, install the required libraries by executing the command below.

```bash
pip install numpy==1.23.5
pip install -r requirements.txt
```

Next, set up LION by executing the command below.

```bash
python setup.py develop
```

Finally, set up Mamba by executing the command below.

```bash
cd pcdet/ops/mamba
python setup.py install
```

## Inference

Run the command below to start the inference.

```bash
python tools/simple_inference.py <input_file_path> <output_file_path> <config_file_path> <checkpoint_file_path> <options>...
```

### Example Executions

#### When using ROSBAG as input

```bash
python tools/simple_inference.py input_dir output_dir config.yaml checkpoint.pth --input_topic_name /lidar0/pandar_points
```

#### When setting output format to ROS1

```bash
python tools/simple_inference.py input_dir output.bag config.yaml checkpoint.pth --output_format ROS1 --output_topic_name /detected_objects
```

### Parameters

#### Required Parameters

| Parameter    | Tag        | Default    | Description         |
|-------------|-------------|-------------|-------------|
| input_dir (必須) | ─       | ─            | Path to the input file. .bin, .npy, .pcd, ROSBAG are supported. |
| output_dir (必須)  | ─     | ─           | Path to the output directory. (Note: When setting the output format to ROS1, use .bag as the extension.) |
| config (必須) | ─       | ─            | Path to the config file. |
| checkpoint (必須)  | ─     | ─           | Path to the checkpoint file. |

#### Optional Parameters

| Parameter    | Tag        | Default    | Description         |
|-------------|-------------|-------------|-------------|
| --help      | -h          | ─           | Show help. |
| --output_format | ─          | CSV   | Set the output format. |
| --score_threshold | ─          | 0.0         | Set the score threshold for controlling the output (0~1). |
| --mapping_json | ─       | ─           | Set the JSON file that contains the mapping of classes to output. If not set, the model's default will be used. |

#### Parameters for Using ROSBAG as Input

| Parameter    | Tag        | Default    | Description         |
|-------------|-------------|-------------|-------------|
| --input_topic_name | ─         | ─           | Set the topic name to use. If the --input_topic_name option is not set, all topics will be used. |

#### Required Parameters When Using ROSBAG as Output

| Parameter    | Tag        | Default    | Description         |
|-------------|-------------|-------------|-------------|
| --output_topic_name (Required in ROSBAG output) | ─        | ─           | Set the topic name to save the inference results. |

### Model

#### Model Performance

| Model | Inference Time | GPU Usage |
| --- | --- | --- |
| LION-RetNet | 0.3367 sec / frame | 2268 MiB |
| LION-Mamba | 0.3165 sec / frame | 2192 MiB |

## Training and Evaluation

