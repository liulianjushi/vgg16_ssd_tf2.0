import tensorflow as tf

from configuration import MAP_SIZE
from nets import dataset
from nets.anchor_layer import Anchor
from nets.vgg_layer import SsdLayer


class IouLayer(object):
    def __init__(self):
        super(IouLayer, self).__init__()
        pass

    def __call__(self, inputs, **kwargs):
        box_1, box_2 = inputs

        box_1_area = box_1[..., 2] * box_1[..., 3]
        box_2_area = box_2[..., 2] * box_2[..., 3]

        intersect_min = tf.maximum(box_1[..., 0:2], box_2[..., 0:2])
        intersect_max = tf.minimum(box_1[..., 2:4], box_2[..., 2:4])
        intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou


class EncodeLayer(tf.keras.layers.Layer):
    def __init__(self,anchor=None):
        super(EncodeLayer, self).__init__()
        # self.map_size = tf.constant(MAP_SIZE)
        # self.anchor = Anchor()
        self.default_boxes = anchor
        self.iou_layer = IouLayer()

    @tf.function
    def get_offset(self, box_true, box_pred):
        # 计算边框的中心点与宽高
        box_true = tf.stack([(box_true[..., 2] + box_true[..., 0]) / 2, (box_true[..., 3] + box_true[..., 1]) / 2,
                             box_true[..., 2] - box_true[..., 0], box_true[..., 3] - box_true[..., 1]], 1)

        box_pred = tf.stack([(box_pred[..., 2] + box_pred[..., 0]) / 2, (box_pred[..., 3] + box_pred[..., 1]) / 2,
                             box_pred[..., 2] - box_pred[..., 0], box_pred[..., 3] - box_pred[..., 1]], 1)

        box_true = box_true - tf.concat(
            [box_pred[..., 0:2], tf.zeros_like(box_pred[..., 2:4])], axis=-1)
        box_true = box_true / tf.tile(box_pred[..., 2:4], (1, 2))
        box_true = tf.concat(
            [box_true[..., 0:2], tf.math.log(box_true[..., 2:4])], axis=-1)

        return box_true

    def call(self, inputs, **kwargs):
        batch_labels = []
        for labels in inputs:
            box_true = tf.boolean_mask(labels, tf.not_equal(labels[..., -1], 0))
            # 映射边框
            box_true_1 = tf.reshape(tf.tile(box_true, [1, self.default_boxes.shape[0]]), (-1, box_true.shape[1]))
            box_pred_1 = tf.tile(self.default_boxes, [box_true.shape[0], 1])
            ious = tf.reshape(self.iou_layer((box_true_1, box_pred_1)), (box_true.shape[0], -1))
            max_index = tf.argmax(ious, axis=0)
            box_pred_assigned = self.get_offset(box_true=tf.gather(box_true, list(max_index.numpy()), 0)[..., :4],
                                                box_pred=self.default_boxes)

            # 映射类别
            iou_max = tf.reduce_max(ious, axis=0)
            max_index_class = tf.gather(box_true, list(max_index.numpy()), 0)[..., -1]
            pos_boolean = tf.where(iou_max >= 0.5, 1.0, 0.0)  # 1 for positive, 0 for negative
            pos_class_index = max_index_class * pos_boolean
            pos_class_index = tf.reshape(pos_class_index, (-1, 1))

            labeled_box_pred = tf.concat((box_pred_assigned, pos_class_index), axis=-1)
            batch_labels.append(labeled_box_pred)
        return tf.stack(batch_labels, 0)


if __name__ == '__main__':
    train_dataset = dataset.VOCDataSet("/Users/james/IdeaProjects/vgg16_ssd_tf2.0/data/val.record", batch_size=4,
                                       epoch=1).load_data()
    num = len(list(train_dataset.as_numpy_iterator()))
    print(f"train dataset num:{num}")

    encode_layer = EncodeLayer()
    ssd = SsdLayer(num_classes=20)
    ssd.build(input_shape=(None, 300, 300, 3))
    for step, (images, labels) in enumerate(train_dataset):
        labs = encode_layer(labels)
