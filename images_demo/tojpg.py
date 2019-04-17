from PIL import Image
import os


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


def transimg(input_file, output_file):
    """
    转换图片格式
    :param img_path:图片路径
    :return: True：成功 False：失败
    """
    if IsValidImage(input_file):
        try:
            im = Image.open(input_file)
            im.save(output_file)
            return True
        except:
            return False
    else:
        return False


if __name__ == '__main__':

    img_path = 'C:/Users/86529/Desktop/images/'
    dest_path = 'C:/Users/86529/Desktop/dest/'
    n = 0
    for root, dirs, files in os.walk(img_path):
        for jpgs in files:
            if transimg(img_path + jpgs, dest_path + str(n) + '.jpg') == False:
                print(files)

            n = n+1