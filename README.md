# 概要

画像中の「人」だけを削除する。

## 学習結果

<img src="https://github.com/ka10ryu1/human_delete/blob/img/concat.jpg" width="640px">

# 動作環境

- **Ubuntu** 16.04.3 LTS ($ cat /etc/issue)
- **Python** 3.5.2 ($ python3 -V)
- **chainer** 3.5 ($ pip3 show chainer | grep Ver)
- **numpy** 1.14.2 ($ pip3 show numpy | grep Ver)
- **cupy** 2.4 ($ pip3 show cupy | grep Ver)
- **opencv-python** 3.4.0.12 ($ pip3 show opencv-python | grep Ver)

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 3 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
$ tree >& log.txt
```

## ファイル

```console
.
├── Image
│   ├── background > 背景画像フォルダ
│   ├── other      > 物体画像フォルダ
│   └── people     > 人物画像フォルダ
├── LICENSE
├── Lib
│   ├── network.py > human_deleteのネットワーク部分
│   └── plot_report_log.py
├── README.md
├── Tools
│   ├── LICENSE
│   ├── README.md
│   ├── Tests
│   │   ├── Lenna.bmp       > テスト用画像
│   │   ├── Mandrill.bmp    > テスト用画像
│   │   ├── test_getfunc.py > getfuncのテスト用コード
│   │   └── test_imgfunc.py > imgfuncのテスト用コード
│   ├── concat.py          > 複数の画像を任意の行列で結合する
│   ├── dot2png.py         > dot言語で記述されたファイルをPNG形式に変換する
│   ├── func.py            > 便利機能
│   ├── getfunc.py         > 画像処理に関する便利機能
│   ├── imgfunc.py         > 画像処理に関する便利機能
│   ├── npz2jpg.py         > 作成したデータセット（.npz）の中身を画像として出力する
│   ├── plot_diff.py       > logファイルの複数比較
│   └── png_monitoring.py  > 任意のフォルダの監視
├── auto_train.sh
├── clean_all.sh
├── create_dataset.py > 画像を読み込んでデータセットを作成する
├── predict.py        > モデルとモデルパラメータを利用して推論実行する
└── train.py          > 学習メイン部
```

# チュートリアル

GPU環境でそこそこepochまわせないとまともな結果にならないので注意。

## データセットを作成する

### 実行

```console
$ ./create_dataset.py
```

### 生成物の確認

以下のファイルが生成されていれば成功

- test_256x256_000020.npz
- train_256x256_000180.npz

## 学習する

### 実行

```console
$ ./train.py
```

### 生成物の確認

以下のファイルがresultフォルダに生成されていれば成功

- loss.png
- *.json
- *.log
- *.model
- *_10.snapshot
- *_graph.dot

## 推論実行

### 実行

```console
$ ./predict.py result/*.model result/*.json
```

### 生成物の確認

以下のファイルがresultフォルダ生成されていれば成功

- predict.jpg

## 推論実行（複数回連続実行）

### 実行

以下を実行することで12枚の入力画像と出力画像を連結した画像が生成される。

```console
$ ./predict_repeat.py result/*.model result/*.json
```

以下を実行することで、生成された12枚の画像を1枚に結合できる。

```console
$ Tools/concat.py ./result/predict-00* -r 3
```

### 生成物の確認

以下のファイルがresultフォルダ生成されていれば成功。

- predict-00**.jpg < 12枚
- concat.jpg
