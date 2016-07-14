#!/usr/bin/env sh
DATA=/opt/caffe/data/flower_photos

echo "Creating lmdb."

rm -rf $DATA/train_lmdb
rm -rf $DATA/val_lmdb

convert_imageset \
    -shuffle \
    -resize_height 256 \
    -resize_width 256 \
    $DATA/ \
    $DATA/train.txt \
    $DATA/train_lmdb 

convert_imageset \
    -shuffle \
    -resize_height 256 \
    -resize_width 256 \
    $DATA/ \
    $DATA/val.txt \
    $DATA/val_lmdb 


echo "Done."




