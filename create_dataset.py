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
    parser.add_argument('jpeg', nargs='+',
                        help='使用する画像のパス')
    parser.add_argument('--channel', '-c', type=int, default=1,
                        help='画像のチャンネル数 [default: 1 channel]')
    parser.add_argument('--augmentation', '-a', type=int, default=2,
                        help='水増しの種類 [default: 2]')
    parser.add_argument('--img_size', '-s', type=int, default=32,
                        help='生成される画像サイズ [default: 32 pixel]')
    parser.add_argument('--round', '-r', type=int, default=1000,
                        help='切り捨てる数 [default: 1000]')
    parser.add_argument('--quality', '-q', type=int, default=5,
                        help='画像の圧縮率 [default: 5]')
    parser.add_argument('--train_per_all', '-t', type=float, default=0.9,
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


def paste(fg, bg):
    # Load two images
    img1 = bg.copy()
    img2, _ = IMG.rotateR(fg, [-90, 90], 1.0)

    # I want to put logo on top-left corner, So I create a ROI
    w1, h1 = img1.shape[:2]
    w2, h2 = img2.shape[:2]
    x = np.random.randint(0, w1 - w2 + 1)
    y = np.random.randint(0, w1 - w2 + 1)
    roi = img1[x:x + w2, y:y + h2]

    # Now create a mask of logo and create its inverse mask also
    mask = img2[:, :, 3]
    ret, mask_inv = cv2.threshold(
        cv2.bitwise_not(mask),
        200, 255, cv2.THRESH_BINARY
    )

    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    mask_inv = cv2.dilate(mask_inv, kernel1, iterations=1)
    mask_inv = cv2.erode(mask_inv, kernel2, iterations=1)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    # Put logo in ROI and modify the main image
    dst = cv2.add(img1_bg, img2_fg)
    img1[x:x + w2, y:y + h2] = dst
    return img1


def create(obj_path, h_path, bg_path, num):
    x = []
    y = []
    for i in range(num):
        objects = getSomeImage('Image/other/', 6, 64)
        human = getImage('Image/people/', 64)
        background = rondom_crop(getImage('Image/background/', 256))

        for j in objects:
            background = paste(j, background)

        x.append(paste(human, background))
        y.append(background)

    return np.array(x), np.array(y)


def main(args):

    print('create images...')
    x, y = create('Image/other/', 'Image/people/', 'Image/background/', 1000)

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
