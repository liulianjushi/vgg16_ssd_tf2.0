import tensorflow as tf

from configuration import MAX_BOXES_PER_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH


class VOCDataSet(object):
    def __init__(self, file="", max_boxes_per_image=MAX_BOXES_PER_IMAGE, batch_size=1, epoch=1):
        self.dataset = tf.data.TFRecordDataset(file, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        self.max_boxes_per_image = max_boxes_per_image
        self.epoch = epoch
        self.batch_size = batch_size

    def __len__(self):
        return len(list(self.dataset.as_numpy_iterator()))

    def decode_and_resize(self, serialized_example):
        feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }

        features = tf.io.parse_single_example(serialized_example, feature_description)
        image = tf.io.decode_jpeg(features["image/encoded"])
        image = tf.image.resize(image, [IMAGE_HEIGHT, IMAGE_WIDTH])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # print(features["image/object/bbox/xmin"])
        xmins = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/xmin"]), (-1, 1))
        xmaxs = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/xmax"]), (-1, 1))
        ymins = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/ymin"]), (-1, 1))
        ymaxs = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/ymax"]), (-1, 1))
        classes = tf.cast(tf.reshape(tf.sparse.to_dense(features["image/object/class/label"]), (-1, 1)),
                          dtype=tf.float32)
        label = tf.concat([xmins, ymins, xmaxs, ymaxs, classes], 1) / [
            features["image/width"], features["image/height"], features["image/width"], features["image/height"], 1]
        a = tf.zeros([self.max_boxes_per_image, label.shape[-1]], tf.float32)
        labels = tf.concat([label, a], 0)[:self.max_boxes_per_image, ...]
        image = {"raw": image, "height": features["image/height"], "width": features["image/width"]}
        return image, labels

    def load_data(self):
        dataset = self.dataset.cache() \
            .map(self.decode_and_resize, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .shuffle(buffer_size=len(list(self.dataset.as_numpy_iterator()))) \
            .batch(self.batch_size) \
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE) \
            .repeat(self.epoch)
        return dataset
