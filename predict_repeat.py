#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'predictを繰り返し実行する'
#

import cv2
import argparse
import numpy as np

import chainer
import chainer.links as L

from Lib.network import JC_DDUU as JC
from create_dataset import create
import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F

from predict import predict


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
    parser.add_argument('-on', '--obj_num', type=int, default=4,
                        help='障害物の最大数 [default: 4]')
    parser.add_argument('-hn', '--human_num', type=int, default=2,
                        help='人間の最大数 [default: 2]')
    parser.add_argument('-in', '--img_num', type=int, default=6,
                        help='画像を生成する数 [default: 6]')
    parser.add_argument('-is', '--img_size', type=int, default=256,
                        help='生成される画像サイズ [default: 256 pixel]')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('--predict_num', '-pn', type=int, default=12,
                        help='推論実行反復回数 [default: 12]')
    parser.add_argument('--wait', '-w', type=int, default=500,
                        help='画像を表示させる待ち時間 [ms] [default: 500]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    return parser.parse_args()


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    net, unit, ch, size, layer, sr, af1, af2 = GET.modelParam(args.param)
    # 学習モデルを生成する
    model = L.Classifier(
        JC(n_unit=unit, n_out=ch, rate=sr, actfun1=af1, actfun2=af2)
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

    for i in range(args.predict_num):
        # 画像の生成
        x, _ = create(args.other_path,
                      args.human_path,
                      args.background_path,
                      args.obj_size, args.img_size,
                      args.obj_num, args.human_num, 1)

        # 学習モデルを実行する
        with chainer.using_config('train', False):
            img = IMG.resize(predict(model, x, args.batch, args.gpu), 1 / sr)

        # 生成結果を保存する
        name = F.getFilePath(args.out_path, 'predict-' +
                             str(i).zfill(4), '.jpg')
        print('save:', name)
        img = np.hstack([x[0], img])
        cv2.imwrite(name, img)
        cv2.imshow('view', img)
        cv2.waitKey(args.wait)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
