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
    parser.add_argument('-in', '--img_num', type=int, default=200,
                        help='画像を生成する数 [default: 200]')
    parser.add_argument('-on', '--obj_num', type=int, default=4,
                        help='障害物の最大数 [default: 4]')
    parser.add_argument('-hn', '--human_num', type=int, default=2,
                        help='人間の最大数 [default: 2]')
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


def getSomeImage(imgs, num, size):
    imgs = np.array(imgs)
    label = range(len(imgs))
    if(num > 1):
        pickup = np.random.choice(
            label, np.random.randint(1, num), replace=False
        )
        if size > 1:
            return [IMG.resizeP(img, size) for img in imgs[pickup]]
        else:
            imgs[pickup]

    else:
        pickup = np.random.choice(label, 1, replace=False)[0]
        if size > 1:
            return IMG.resizeP(imgs[pickup], size)
        else:
            return imgs[pickup]


def rondom_crop(img, size):
    w, h = img.shape[: 2]
    short_side = min(img.shape[: 2])
    x = np.random.randint(0, w - short_side + 1)
    y = np.random.randint(0, h - short_side + 1)
    if size > 1:
        return IMG.resizeP(img[x: x + short_side, y: y + short_side], size)
    else:
        img[x: x + short_side, y: y + short_side]


def getImage(imgs, size):
    return getSomeImage(imgs, 1, size)


def getImgs(path):
    return [cv2.imread(os.path.join(path, f), IMG.getCh(0))
            for f in os.listdir(path)]


def create(obj_path, h_path, bg_path,
           obj_size, img_size,
           obj_num, h_num, create_num):
    obj = getImgs(obj_path)
    hum = getImgs(h_path)
    bg = getImgs(bg_path)

    x = []
    y = []
    for i in range(create_num):
        objects = getSomeImage(obj, obj_num + 1, obj_size)
        human = getSomeImage(hum, h_num + 1, obj_size)
        background = rondom_crop(getImage(bg, -1), img_size)

        for j in objects:
            background = IMG.paste(j, background)

        y.append(background[:, :, :3])
        for k in human:
            background = IMG.paste(k, background)

        x.append(background[:, :, :3])

    return np.array(x), np.array(y)


def main(args):

    print('create images...')
    x, y = create(args.other_path,
                  args.human_path,
                  args.background_path,
                  args.obj_size, args.img_size,
                  args.obj_num, args.human_num, args.img_num)

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
