import tensorflow as tf

from configuration import CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, MAP_SIZE
from nets.anchor_layer import Anchor


class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DecodeLayer, self).__init__()
        self.map_size = tf.constant(MAP_SIZE)
        self.anchor = Anchor()
        self.default_boxes = self.anchor(self.map_size)
    
    
    @tf.function
    def get_offset(self, predict_loc, box_pred):
        # 计算边框的中心点与宽高
        box_pred = tf.stack([(box_pred[..., 2] + box_pred[..., 0]) / 2, (box_pred[..., 3] + box_pred[..., 1]) / 2,
                             box_pred[..., 2] - box_pred[..., 0], box_pred[..., 3] - box_pred[..., 1]], 1)

        predict_loc = predict_loc * tf.concat(
            [box_pred[..., 2:4], tf.ones_like(box_pred[..., 2:4])], axis=-1)
        predict_loc = predict_loc + tf.concat(
            [box_pred[..., 0:2], tf.zeros_like(box_pred[..., 2:4])], axis=-1)
        boxes = tf.concat(
            [predict_loc[..., 0:2], tf.math.exp(predict_loc[..., 2:4]) * box_pred[..., 2:4]], axis=-1)

        boxes = tf.concat([boxes[..., 0:2] - boxes[..., 2:4] / 2, boxes[..., 0:2] + boxes[..., 2:4] / 2], axis=-1)
        return tf.clip_by_value(boxes, 0.0, 1.0)

    def call(self, logits, **kwargs):
        detection_boxes = []
        detection_classes = []
        detection_scores = []
        for logit in logits:
            predict_cls = logit[..., 4:]
            predict_loc = logit[..., :4]
            predict_loc = self.get_offset(predict_loc, self.default_boxes)

            predict_cls = tf.nn.softmax(predict_cls, axis=-1)
            scores = tf.reduce_max(predict_cls, axis=-1)
            classes = tf.argmax(predict_cls, axis=-1)
            mask = tf.not_equal(classes, 0)
            # print("mask:", mask)
            box = tf.boolean_mask(predict_loc, mask=mask, axis=0)
            cls = tf.boolean_mask(classes, mask=mask, axis=0)
            score = tf.boolean_mask(scores, mask=mask, axis=0)
            selected_indices = tf.image.non_max_suppression(box, score, max_output_size=5,
                                                            iou_threshold=NMS_IOU_THRESHOLD,
                                                            score_threshold=CONFIDENCE_THRESHOLD)
            # print("selected_indices:", selected_indices)
            detection_boxes.append(tf.gather(box, selected_indices))
            detection_classes.append(
                tf.cast(tf.expand_dims(tf.gather(cls, selected_indices), axis=-1),
                        dtype=tf.float32))
            detection_scores.append(
                tf.expand_dims(tf.gather(score, selected_indices), axis=-1))
        result = {"detection_boxes": detection_boxes, "detection_classes": detection_classes,
                  "detection_scores": detection_scores}
        return result
