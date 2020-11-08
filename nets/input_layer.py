import tensorflow as tf

from configuration import MAX_BOXES_PER_IMAGE

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


class InputLayer(object):
    def __init__(self, max_boxes_per_image=MAX_BOXES_PER_IMAGE, batch_size=1, epoch=1, is_train=True):
        super(InputLayer, self).__init__()
        self.max_boxes_per_image = max_boxes_per_image
        self.is_train = is_train
        self.epoch = epoch
        self.batch_size = batch_size

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
        image = tf.image.resize(image, [300, 300])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

        # print(features["image/object/bbox/xmin"])
        xmins = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/xmin"]), (-1, 1))
        xmaxs = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/xmax"]), (-1, 1))
        ymins = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/ymin"]), (-1, 1))
        ymaxs = tf.reshape(tf.sparse.to_dense(features["image/object/bbox/ymax"]), (-1, 1))
        classes = tf.cast(tf.reshape(tf.sparse.to_dense(features["image/object/class/label"]), (-1, 1)),
                          dtype=tf.float32)
        # if self.is_train:
        #     label = tf.concat([(xmins + xmaxs) / 2, (ymins + ymaxs) / 2, xmaxs - xmins, ymaxs - ymins, classes], 1) / [
        #         features["image/width"], features["image/height"], features["image/width"], features["image/height"], 1]
        # else:
        label = tf.concat([xmins, ymins, xmaxs, ymaxs, classes], 1) / [
            features["image/width"], features["image/height"], features["image/width"], features["image/height"], 1]
        a = tf.zeros([self.max_boxes_per_image, label.shape[-1]], tf.float32)
        labels = tf.concat([label, a], 0)[:self.max_boxes_per_image, ...]
        image = {"raw": image, "height": features["image/height"], "width": features["image/width"]}
        return image, labels

    def __call__(self, inputs):
        dataset = tf.data.TFRecordDataset(inputs, num_parallel_reads=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(map_func=self._decode_and_resize)
        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(len(list(dataset.as_numpy_iterator())))
        dataset = dataset.repeat(self.epoch)
        dataset = dataset.batch(self.batch_size)
        return dataset
