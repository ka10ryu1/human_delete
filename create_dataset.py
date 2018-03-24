#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import os
import cv2
import argparse
import numpy as np

import Tools.imgfunc as IMG
import Tools.func as F


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-ot', '--other_path', default='./Image/other/',
                        help='・ (default: ./Image/other/')
    parser.add_argument('-hu', '--human_path', default='./Image/people/',
                        help='・ (default: ./Image/people/')
    parser.add_argument('-bg', '--background_path', default='./Image/background/',
                        help='・ (default: ./Image/background/')
    parser.add_argument('-os', '--obj_size', type=int, default=64,
                        help='挿入する画像サイズ [default: 64 pixel]')
    parser.add_argument('-is', '--img_size', type=int, default=256,
                        help='生成される画像サイズ [default: 256 pixel]')
    parser.add_argument('-r', '--round', type=int, default=1000,
                        help='切り捨てる数 [default: 1000]')
    parser.add_argument('-in', '--img_num', type=int, default=1000,
                        help='画像を生成する数 [default: 1000]')
    parser.add_argument('-on', '--oth_num', type=int, default=6,
                        help='画像を生成する数 [default: 6]')
    parser.add_argument('-t', '--train_per_all', type=float, default=0.9,
                        help='画像数に対する学習用画像の割合 [default: 0.9]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    return parser.parse_args()


def saveNPZ(x, y, name, folder, size):
    """
    入力データと正解データをNPZ形式で保存する
    [in] x:      保存する入力データ
    [in] y:      保存する正解データ
    [in] name:   保存する名前
    [in] folder: 保存するフォルダ
    [in] size:   データ（正方形画像）のサイズ
    """

    size_str = '_' + str(size).zfill(2) + 'x' + str(size).zfill(2)
    num_str = '_' + str(x.shape[0]).zfill(6)
    np.savez(F.getFilePath(folder, name + size_str + num_str), x=x, y=y)


def getSomeImage(path, num, size):
    all_path = [os.path.join(path, f) for f in os.listdir(path)]
    if(num > 1):
        img_num = np.random.choice(
            range(len(all_path)), np.random.randint(1, num), replace=False
        )
        return [IMG.resizeP(cv2.imread(all_path[i], IMG.getCh(0)), size)
                for i in img_num]
    else:
        img_num = np.random.choice(range(len(all_path)), 1, replace=False)[0]
        return IMG.resizeP(cv2.imread(all_path[img_num], IMG.getCh(0)), size)


def rondom_crop(img):
    w, h = img.shape[:2]
    short_side = min(img.shape[:2])
    x = np.random.randint(0, w - short_side + 1)
    y = np.random.randint(0, h - short_side + 1)
    return img[x:x + short_side, y:y + short_side]


def getImage(path, size):
    return getSomeImage(path, 1, size)


def create(obj_path, h_path, bg_path, num):
    x = []
    y = []
    for i in range(args.img_num):
        objects = getSomeImage(args.other_path, args.oth_num, args.obj_size)
        human = getImage(args.human_path, args.obj_size)
        background = rondom_crop(getImage(args.background_path, args.img_size))

        for j in objects:
            background = IMG.paste(j, background)

        x.append(IMG.paste(human, background)[:, :, :3])
        y.append(background[:, :, :3])

    return np.array(x), np.array(y)


def main(args):

    print('create images...')
    x, y = create('Image/other/', 'Image/people/',
                  'Image/background/', args.img_num)

    # 画像の並び順をシャッフルするための配列を作成する
    # compとrawの対応を崩さないようにシャッフルしなければならない
    # また、train_sizeで端数を切り捨てる
    # データをint8からfloat16に変えるとデータ数が大きくなるので注意
    print('shuffle images...')
    dtype = np.float16
    shuffle = np.random.permutation(range(len(x)))
    train_size = int(len(x) * args.train_per_all)
    train_x = IMG.imgs2arr(x[shuffle[:train_size]], dtype=dtype)
    train_y = IMG.imgs2arr(y[shuffle[:train_size]], dtype=dtype)
    test_x = IMG.imgs2arr(x[shuffle[train_size:]], dtype=dtype)
    test_y = IMG.imgs2arr(y[shuffle[train_size:]], dtype=dtype)
    print('train x/y:{0}/{1}'.format(train_x.shape, train_y.shape))
    print('test  x/y:{0}/{1}'.format(test_x.shape, test_y.shape))

    # 生成したデータをnpz形式でデータセットとして保存する
    # ここで作成したデータの中身を確認する場合はnpz2jpg.pyを使用するとよい
    print('save npz...')
    saveNPZ(train_x, train_y, 'train', args.out_path, args.img_size)
    saveNPZ(test_x, test_y, 'test', args.out_path, args.img_size)


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
