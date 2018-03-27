#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'モデルとモデルパラメータを利用して推論実行する'
#

import cv2
import time
import argparse
import numpy as np

import chainer
import chainer.links as L
from chainer.cuda import to_cpu

from Lib.network import JC_DDUU as JC
from create_dataset import create
import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('-ot', '--other_path', default='./Image/other/',
                        help='・ (default: ./Image/other/')
    parser.add_argument('-hu', '--human_path', default='./Image/people/',
                        help='・ (default: ./Image/people/')
    parser.add_argument('-bg', '--background_path', default='./Image/background/',
                        help='・ (default: ./Image/background/')
    parser.add_argument('-os', '--obj_size', type=int, default=64,
                        help='挿入する画像サイズ [default: 64 pixel]')
    parser.add_argument('-on', '--obj_num', type=int, default=6,
                        help='画像を生成する数 [default: 6]')
    parser.add_argument('-is', '--img_size', type=int, default=256,
                        help='生成される画像サイズ [default: 256 pixel]')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    return parser.parse_args()


def predict(model, img, batch, gpu):
    """
    推論実行メイン部
    [in]  model:     推論実行に使用するモデル
    [in]  img:       入力画像
    [in]  batch:     バッチサイズ
    [in]  gpu:       GPU ID
    [out] img:       推論実行で得られた生成画像
    """

    # dataには圧縮画像と分割情報が含まれているので、分離する
    st = time.time()
    # バッチサイズごとに学習済みモデルに入力して画像を生成する
    x = IMG.imgs2arr(img, gpu=gpu)
    y = model.predictor(x)
    predict_img = IMG.arr2imgs(to_cpu(y.array))[0]
    print('exec time: {0:.2f}[s]'.format(time.time() - st))
    return predict_img


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    net, unit, ch, size, layer, sr, af1, af2 = GET.modelParam(args.param)
    # 学習モデルを生成する
    model = L.Classifier(
        JC(n_unit=unit, n_out=ch,
           rate=sr, actfun_1=af1, actfun_2=af2)
    )

    # load_npzのpath情報を取得し、学習済みモデルを読み込む
    load_path = F.checkModelType(args.model)
    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(F.fileFuncLine())
        exit()

    # GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # 画像の生成
    x, _ = create(args.other_path,
                  args.human_path,
                  args.background_path,
                  args.obj_size, args.img_size,
                  args.obj_num, 1, 1)

    # 学習モデルを実行する
    with chainer.using_config('train', False):
        img = IMG.resize(predict(model, x, args.batch, args.gpu), 0.5)

    # 生成結果を保存する
    name = F.getFilePath(args.out_path, 'predict', '.jpg')
    print('save:', name)
    img = np.hstack([x[0], img])
    cv2.imwrite(name, img)
    cv2.imshow(name, img)
    cv2.waitKey()


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
