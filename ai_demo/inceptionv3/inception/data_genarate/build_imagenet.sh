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
find . -name "*.tar" |
 while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done
cd .. && cd .. && cd .. && cd ..


python preprocess_imagenet_validation_data.py /home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/validation/ data/imagenet_2012_validation_synset_labels.txt

python process_bounding_boxes.py /home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/bounding_boxes/ data/imagenet_lsvrc_2015_synsets.txt | sort > /home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet_2012_bounding_boxes.csv

python build_imagenet_data.py --train_directory=/home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/train/ --validation_directory=/home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet-data/validation/ --output_directory=/home/antutu/ai/dateset/imagenet/ILSVRC2012/ --imagenet_metadata_file=data/imagenet_metadata.txt --labels_file=data/imagenet_lsvrc_2015_synsets.txt --bounding_box_file=/home/antutu/ai/dateset/imagenet/ILSVRC2012/raw-data/imagenet_2012_bounding_boxes.csv

