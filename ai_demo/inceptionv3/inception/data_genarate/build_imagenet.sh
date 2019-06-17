#!/usr/bin/env bash

mkdir -p ILSVRC2012/raw-data/imagenet-data/bounding_boxes

# 解压boxes
mv ILSVRC2012_bbox_train_v2.tar.gz ILSVRC2012/raw-data/imagenet-data/bounding_boxes/
tar xvf ILSVRC2012/raw-data/imagenet-data/bounding_boxes/ILSVRC2012_bbox_train_v2.tar.gz -C ILSVRC2012/raw-data/imagenet-data/bounding_boxes/
NUM_XML=$(ls -1 ILSVRC2012/raw-data/imagenet-data/bounding_boxes/* | wc -l)
echo "Identified ${NUM_XML} bounding box annotations."

# 解压验证集
mkdir -p ILSVRC2012/raw-data/imagenet-data/validation/
tar xvf ILSVRC2012_img_val.tar -C ILSVRC2012/raw-data/imagenet-data/validation/

# 解压训练集
mkdir -p ILSVRC2012/raw-data/imagenet-data/train/
mv ILSVRC2012_img_train.tar ILSVRC2012/raw-data/imagenet-data/train/ && cd ILSVRC2012/raw-data/imagenet-data/train/
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done
cd .. && cd .. && cd .. && cd ..

# 在argv[1]下创建1000个分类文件夹(argv[2])，并把 validation 图片分别放入对应的类别文件夹
python preprocess_imagenet_validation_data.py /home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/validation/ data/imagenet_2012_validation_synset_labels.txt

# 根据argv[1]目录下的xml文件并根据argv[2]的分类校验是否存在， 生成每张图片的详细信息到文件 raw-data/imagenet_2012_bounding_boxes.csv
python process_bounding_boxes.py /home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/bounding_boxes/ data/imagenet_lsvrc_2015_synsets.txt | sort > /home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet_2012_bounding_boxes.csv

"""
argv :
    --validation_directory      验证集图片所在路径   分类/图片
    --train_directory           训练集图片所在路径   分类/图片
    --output_directory          生成TFRecord文件所在路径
    --imagenet_metadata_file    分类标签数字标识与人类可识别标签对应文件
    --labels_file               分类标签数字标识
    --bounding_box_file         图片给详情与对象边界信息
"""
python build_imagenet_data.py --validation_directory=/home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/validation/ --train_directory=/home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/train/ --output_directory=/home/antutu/ai/dateset/imagenet/ILSVRC2012/ --imagenet_metadata_file=data/imagenet_metadata.txt --labels_file=data/imagenet_lsvrc_2015_synsets.txt --bounding_box_file=/home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet_2012_bounding_boxes.csv

