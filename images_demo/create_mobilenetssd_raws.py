#
# Copyright (c) 2016,2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import numpy as np
import os
import shutil

from PIL import Image


def IsValidImage(img_path):
    """
    判断文件是否为有效（完整）的图片
    :param img_path:图片路径
    :return:True：有效 False：无效
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path, dest_image):
    """
    转换图片格式
    :param img_path:图片路径
    :return: True：成功 False：失败
    """
    if IsValidImage(img_path):
        try:
            im = Image.open(img_path)
            im.save(dest_image)
            return True
        except:
            return False
    else:
        return False


def __get_img_raw(img_filepath):
    img_filepath = os.path.abspath(img_filepath)
    img = Image.open(img_filepath)
    img_ndarray = np.array(img) # read it
    if len(img_ndarray.shape) != 3:
        raise RuntimeError('Image shape' + str(img_ndarray.shape))
    if (img_ndarray.shape[2] != 3):
        raise RuntimeError('Require image with rgb but channel is %d' % img_ndarray.shape[2])
    # reverse last dimension: rgb -> bgr
    return img_ndarray


def __create_mean_raw(img_raw, mean_rgb):
    if img_raw.shape[2] != 3:
        raise RuntimeError('Require image with rgb but channel is %d' % img_raw.shape[2])
    img_dim = (img_raw.shape[0], img_raw.shape[1])
    mean_raw_r = np.empty(img_dim)
    mean_raw_r.fill(mean_rgb[0])
    mean_raw_g = np.empty(img_dim)
    mean_raw_g.fill(mean_rgb[1])
    mean_raw_b = np.empty(img_dim)
    mean_raw_b.fill(mean_rgb[2])
    # create with c, h, w shape first
    tmp_transpose_dim = (img_raw.shape[2], img_raw.shape[0], img_raw.shape[1])
    mean_raw = np.empty(tmp_transpose_dim)
    mean_raw[0] = mean_raw_r
    mean_raw[1] = mean_raw_g
    mean_raw[2] = mean_raw_b
    # back to h, w, c
    mean_raw = np.transpose(mean_raw, (1, 2, 0))
    return mean_raw.astype(np.float32)


def __create_raw_incv3(img_filepath, mean_rgb, div, req_bgr_raw, save_uint8):
    img_raw = __get_img_raw(img_filepath)
    mean_raw = __create_mean_raw(img_raw, mean_rgb)

    snpe_raw = img_raw - mean_raw
    snpe_raw = snpe_raw.astype(np.float32)
    # scalar data divide
    snpe_raw /= div

    if req_bgr_raw:
        snpe_raw = snpe_raw[..., ::-1]

    if save_uint8:
        snpe_raw = snpe_raw.astype(np.uint8)
    else:
        snpe_raw = snpe_raw.astype(np.float32)

    img_filepath = os.path.abspath(img_filepath)
    filename, ext = os.path.splitext(img_filepath)
    snpe_raw_filename = filename
    snpe_raw_filename += '.raw'
    snpe_raw.tofile(snpe_raw_filename)

    return 0


def __resize_square_to_jpg(src, dst, size):
    src_img = Image.open(src)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2: src_img = src_img.convert(mode = 'RGB')
    # center crop to square
    width, height = src_img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    img = src_img.crop(crop_coord)
    # resize to alexnet size
    dst_img = img.resize((size, size), Image.ANTIALIAS)
    # save output - save determined from file extension
    dst_img.save(dst)
    return 0


def convert_img(src,dest, txt, size):
    print("Converting images for inception v3 network.")

    print("Converting images for jpg.")
    for root,dirs,files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            str = src_image.rsplit(".", 1)
            des_image = str[0] + ".jpg"
            transimg(src_image, des_image)

    print("Scaling to square: " + src)
    for root,dirs,files in os.walk(src):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' in src_image):
                print(src_image)
                dest_image = os.path.join(dest, jpgs)
                __resize_square_to_jpg(src_image,dest_image,size)

    print("Image mean: " + dest)
    for root,dirs,files in os.walk(dest):
        for jpgs in files:
            src_image=os.path.join(root, jpgs)
            if('.jpg' in src_image):
                print(src_image)
                mean_rgb=(128,128,128)
                __create_raw_incv3(src_image,mean_rgb,128,False,False)

    with open(txt, 'w') as f:
        file_list = []
        for root, dirs, files in os.walk(dest):
            for file in files:
                if ('.raw' in file):
                    list.append(file_list + "\n")
        f.writelines(file_list)


def main():
    parser = argparse.ArgumentParser(description="Batch convert jpgs",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dest',type=str, default='C:/Users/86529/Desktop/data')
    parser.add_argument('-s','--size',type=int, default=300)
    parser.add_argument('-i','--img_folder',type=str, default='C:/Users/86529/Desktop/data2')

    args = parser.parse_args()

    size = args.size
    src = os.path.abspath(args.img_folder)
    dest = os.path.abspath(args.dest)

    if os.path.exists(dest):
        shutil.rmtree(dest)

    os.mkdir(dest)

    convert_img(src,dest,'C:/Users/86529/Desktop/data/raw_list.txt',size)

if __name__ == '__main__':
    exit(main())
