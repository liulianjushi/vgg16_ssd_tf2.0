import tensorflow as tf


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
        return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

    def __call__(self, inputs, **kwargs):
        box_1, box_2 = inputs

        box_1_min, box_1_max = self.__get_box_min_and_max(box_1)
        box_2_min, box_2_max = self.__get_box_min_and_max(box_2)

        box_1_area = self.__get_box_area(box_1)
        box_2_area = self.__get_box_area(box_2)

        # y1 = tf.maximum(box_1[..., 0], box_2[..., 0])
        # y2 = tf.minimum(box_1[..., 2], box_2[..., 2])
        # x1 = tf.maximum(box_1[..., 1], box_2[..., 1])
        # x2 = tf.minimum(box_1[..., 3], box_2[..., 3])
        # intersect_area = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

        intersect_min = tf.maximum(box_1_min, box_2_min)
        intersect_max = tf.minimum(box_1_max, box_2_max)
        intersect_wh = tf.maximum(intersect_max - intersect_min, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        union_area = box_1_area + box_2_area - intersect_area
        iou = intersect_area / union_area
        return iou


class EncodeLayer(tf.keras.layers.Layer):
    def __init__(self, anchor=None):
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
        IOU = []
        for labels in inputs:
            box_true = tf.boolean_mask(labels, tf.not_equal(labels[..., -1], 0))
            # box_true=labels
            # 映射边框
            # 填充背景边框，设为[0,0,0,0,0] 分别为[xmin,ymin,xmax,ymax,class]
            box_true = tf.pad(box_true, [[1, 0], [0, 0]])
            box_true_1 = tf.reshape(tf.tile(box_true, [1, self.default_boxes.shape[0]]), (-1, box_true.shape[1]))
            box_pred_1 = tf.tile(self.default_boxes, [box_true.shape[0], 1])
            ious = tf.reshape(self.iou_layer((box_true_1, box_pred_1)), (box_true.shape[0], -1))
            max_index = tf.argmax(ious)
            iou_max = tf.reduce_max(ious, axis=0)
            # 1 for positive, 0 for negative
            pos_boolean = tf.where(iou_max > 0.5, 1.0, 0.0)

            max_index *= tf.cast(pos_boolean, tf.int64)

            max_index_box_true = tf.gather(box_true, max_index)
            # b = tf.zeros_like(max_index_box_true[..., : -1], tf.float32)
            temp_box = []
            for i in tf.range(max_index_box_true.shape[0]):
                if max_index_box_true[i, -1] == 0:
                    temp_box.append(self.default_boxes[i])
                else:
                    temp_box.append(max_index_box_true[i][:-1])
            box_pred_assigned = self.get_offset(box_true=tf.stack(temp_box, axis=0), box_pred=self.default_boxes)

            # 映射类别
            max_index_class = max_index_box_true[..., -1]
            pos_class_index = max_index_class * pos_boolean
            pos_class_index = tf.reshape(pos_class_index, (-1, 1))

            labeled_box_pred = tf.concat((box_pred_assigned, pos_class_index), axis=-1)
            batch_labels.append(labeled_box_pred)
            IOU.append(ious)
        return tf.stack(batch_labels, 0), IOU

