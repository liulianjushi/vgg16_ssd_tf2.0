import tensorflow as tf

from configuration import MAP_SIZE
from nets.anchor_layer import Anchor
from nets.input_layer import InputLayer
from nets.vgg_layer import SsdLayer


class IouLayer(object):
    def __init__(self):
        super(IouLayer, self).__init__()
        pass

    @staticmethod
    def __get_box_min_and_max(box):
        box_min = box[..., 0:2]
        box_max = box[..., 2:4]
        # box_min = box_xy - box_wh / 2
        # box_max = box_xy + box_wh / 2
        return box_min, box_max

    @staticmethod
    def __get_box_area(box):
        return box[..., 2] * box[..., 3]

    def __call__(self, inputs, **kwargs):
        box_1, box_2 = inputs
        box_1_min, box_1_max = self.__get_box_min_and_max(box_1)
        box_2_min, box_2_max = self.__get_box_min_and_max(box_2)
        box_1_area = self.__get_box_area(box_1)
        box_2_area = self.__get_box_area(box_2)

        # xmin = tf.reduce_max(box_1_min[..., 0], box_2_min[..., 0],axis=0)
        # ymin = tf.reduce_min(box_1_min[..., 1], box_2_min[..., 1])
        #
        # xmax = tf.reduce_min(box_1_max[..., 0], box_2_max[..., 0])
        # ymax = tf.reduce_max(box_1_max[..., 1], box_2_max[..., 1])
        #
        # h = tf.maximum(ymax - ymin, 0)
        # w = tf.maximum(xmax - xmin, 0)
        # intersect_area = w * h

        intersect_min = tf.maximum(box_1_min, box_2_min)
        intersect_max = tf.minimum(box_1_max, box_2_max)
        intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou


class EncodeLayer(object):
    def __init__(self):
        super(EncodeLayer, self).__init__()
        self.map_size = tf.constant(MAP_SIZE)
        self.anchor = Anchor()
        self.default_boxes = self.anchor(self.map_size)
        self.iou_layer = IouLayer()

    def __get_offset(self, box_true, box_pred):
        
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

    def __call__(self, inputs, **kwargs):
        batch_labels = []
        for labels in inputs:
            box_true = tf.boolean_mask(labels, tf.not_equal(labels[..., -1], 0))
            # 映射边框
            box_true_1 = tf.reshape(tf.tile(box_true, [1, self.default_boxes.shape[0]]), (-1, box_true.shape[1]))
            box_pred_1 = tf.tile(self.default_boxes, [box_true.shape[0], 1])
            ious = tf.reshape(self.iou_layer((box_true_1, box_pred_1)), (box_true.shape[0], -1))
            max_index = tf.argmax(ious, axis=0)
            box_pred_assigned = self.__get_offset(box_true=tf.gather(box_true, list(max_index.numpy()), 0)[..., :4],
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
    input = InputLayer(batch_size=4)
    dataset = input("/Users/james/IdeaProjects/vgg16_ssd_tf2.0/data/val.record")
    encode_layer = EncodeLayer()

    ssd = SsdLayer(num_classes=20)
    ssd.build(input_shape=(None, 300, 300, 3))
    for step, (images, labels) in enumerate(dataset):
        labels = encode_layer(labels).numpy()
        ssd(images)
        print(ssd.logits.shape, labels.shape)
