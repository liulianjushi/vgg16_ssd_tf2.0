"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import io
import os
import xml.etree.ElementTree as ET
from collections import namedtuple

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    try:
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)



VOC_BBOX_LABEL_NAMES = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


class GenerateTFRecord(object):
    def __init__(self, split_rate=0.7):
        self.labels_csv = "data/labels.csv"
        self.train_csv = "data/train_labels.csv"
        self.train_record = "data/train.record"
        self.test_csv = "data/val_labels.csv"
        self.test_record = "data/val.record"
        self.split_rate = split_rate

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(self, group, path):
        # with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        #     encoded_jpg = fid.read()
        encoded_jpg = open(os.path.join(path, '{}'.format(group.filename)), 'rb').read()
        height, width, channels = tf.image.decode_jpeg(encoded_jpg).shape
        print(os.path.join(path, '{}'.format(group.filename)), (height, width, channels))
        # encoded_jpg_io = io.BytesIO(encoded_jpg)
        # tf.image.decode_jpeg()
        # image = Image.open(encoded_jpg_io)
        # width, height = image.size

        # image = Image.open(os.path.join(path, '{}'.format(group.filename)))
        # encoded_jpg=image.to
        # width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        labels = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'])
            xmaxs.append(row['xmax'])
            ymins.append(row['ymin'])
            ymaxs.append(row['ymax'])
            classes_text.append(row['name'].encode('utf8'))
            labels.append(row['class'])

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        }))
        return tf_example

    def xml_to_csv(self, path):

        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            # print(xml_file)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            if root.findall('object') is not None:
                for member in root.findall('object'):
                    bndbox = member.find("bndbox")
                    xmin = int(bndbox[0].text)
                    ymin = int(bndbox[1].text)
                    xmax = int(bndbox[2].text)
                    ymax = int(bndbox[3].text)

                    width = int(root.find('size')[0].text)
                    height = int(root.find('size')[1].text)
                    name = member[0].text

                    if xmin > width or ymin > height or xmax < 0 or ymax < 0:
                        continue

                    xmin = 0 if xmin < 0 else xmin
                    ymin = 0 if ymin < 0 else ymin
                    xmax = width if xmax > width else xmax
                    ymax = height if ymax > height else ymax

                    value = (
                        root.find('filename').text, width, height, name, VOC_BBOX_LABEL_NAMES.index(name)+1,
                        member[3].text, ymin,xmin, ymax, xmax)
                    xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'name', 'class', 'difficult', 'ymin', 'xmin', 'ymax', 'xmax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv(self.labels_csv, index=False)
        print('Successfully converted xml to csv.')

    def split_labels(self, rate):
        full_labels = pd.read_csv(self.labels_csv)
        gb = full_labels.groupby('filename')
        grouped_list = [gb.get_group(x) for x in gb.groups]
        total = len(grouped_list)
        train_index = np.random.choice(len(grouped_list), size=int(total * rate), replace=False)
        test_index = np.setdiff1d(list(range(total)), train_index)
        if len(train_index) != 0:
            train = pd.concat([grouped_list[i] for i in train_index])
            train_num = train.groupby("name").size().reset_index(name='number')
            train.to_csv(self.train_csv, index=False)
            print("Number of train dataset is:\n ", train_num)
        if len(test_index) != 0:
            test = pd.concat([grouped_list[i] for i in test_index])
            val_num = test.groupby("name").size().reset_index(name='number')
            test.to_csv(self.test_csv, index=False)
            print("Number of val dataset is:\n", val_num)

    def generate_tfrecord(self, images_path, csv_input, output_path):
        print(images_path)
        print(csv_input)
        writer = tf.io.TFRecordWriter(output_path)
        examples = pd.read_csv(csv_input)
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, images_path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        print('Successfully created the TFRecords: {}'.format(output_path))

    def make_data(self, images_path, annotations_path, rate,data_type = ["train","test"]):
        
        self.xml_to_csv(annotations_path)
        self.split_labels(rate)
        
        if "train" in data_type:
            self.generate_tfrecord(images_path, self.train_csv, self.train_record)
        if "test" in data_type:
            self.generate_tfrecord(images_path, self.test_csv, self.test_record)


if __name__ == '__main__':
    annotations_path = "/root/workspace/data/VOCdevkit/VOC2007/test/Annotations"
    images_path = "/root/workspace/data/VOCdevkit/VOC2007/test/JPEGImages"

    generate_data = GenerateTFRecord()
    generate_data.make_data(images_path, annotations_path, 0.0,["test"])
