<!-- markdownlint-disable MD024 -->
<!-- markdownlint-disable MD033 -->
<!-- markdownlint-disable MD041 -->

[English README is here](https://github.com/MapIV/LION/tree/main/tools/README.md)

# LION

## インストール方法

### condaによるインストール

まずは、下のコマンドを実行してconda環境を作成します。

```bash
mconda create -n LION python==3.9
conda activate LION
```

次に、下のコマンドを実行してPytorchをインストールします。

```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

次に、下のコマンドを実行してライブラリをインストールします。

```bash
pip install numpy==1.23.5
pip install -r requirements.txt
```

次に、下のコマンドを実行してLIONをセットアップします。

```bash
python setup.py develop
```

最後に、下のコマンドを実行してMambaのセットアップをします。

```bash
cd pcdet/ops/mamba
python setup.py install
```

## 推論方法

下のコマンドを実行して推論を開始します。

```bash
python tools/simple_inference.py <入力ファイルパス> <出力ファイルパス> <configファイルパス>　<checkpointファイルパス> <オプション>...
```

### 実行例

#### ROSBAGを入力とする場合

```bash
python tools/simple_inference.py input_dir output_dir config.yaml checkpoint.pth --input_topic_name /lidar0/pandar_points
```

#### 出力形式をROS1にする場合

```bash
python tools/simple_inference.py input_dir output.bag config.yaml checkpoint.pth --output_format ROS1 --output_topic_name /detected_objects
```

### パラメータ

#### 必須パラメータ

| パラメータ    | タグ        | デフォルト    | 説明         |
|-------------|-------------|-------------|-------------|
| input_dir (必須) | ─       | ─            | 入力ファイルのパスを設定します。.bin, .npy, .pcd, ROSBAGがサポートされています。 |
| output_dir (必須)  | ─     | ─           | 出力ディレクトリのパスを設定します。(注意: 出力形式としてROS1を設定する場合は.bagを拡張子に設定してください。) |
| config (必須) | ─       | ─            | configファイルのパスを設定します。 |
| checkpoint (必須)  | ─     | ─           | checkpointファイルのパスを設定します。|

#### 常時使えるパラメータ

| パラメータ    | タグ        | デフォルト    | 説明         |
|-------------|-------------|-------------|-------------|
| --help      | -h          | ─           | ヘルプを表示します。 |
| --output_format | ─          | CSV   | 出力形式を設定します。 |
| --score_threshold | ─          | 0.0         | 出力を制御するためのスコアのしきい値(0~1)を設定します。 |
| --mapping_json | ─       | ─           | 出力するクラスのマッピングが書かれたjsonファイルを設定します。設定されなかった場合はモデルのデフォルトで出力されます。 |

#### 入力としてROSBAGを渡すときに使えるパラメータ

| パラメータ    | タグ        | デフォルト    | 説明         |
|-------------|-------------|-------------|-------------|
| --input_topic_name | ─         | ─           | 使用するトピック名を設定します。--input_topic_nameオプションを設定しない場合は、すべてのトピックを使用します。 |

#### 出力形式としてROSBAGを使用するときに必須のパラメータ

| パラメータ    | タグ        | デフォルト    | 説明         |
|-------------|-------------|-------------|-------------|
| --output_topic_name (ROSBAG出力のときは必須) | ─        | ─           | 推論結果を保存するトピック名を設定します。　|

### モデル

#### モデル性能

| モデル名 | 推論時間 | GPU 使用量 |
| --- | --- | --- |
| LION-RetNet | 0.3367 秒 / フレーム | 2268 MiB |
| LION-Mamba | 0.3165 秒 / フレーム | 2192 MiB |

## 訓練・評価方法

