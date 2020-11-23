import tensorflow as tf

from configuration import CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD, MAX_BOXES_PER_IMAGE, NUM_CLASSES


class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self, anchor=None):
        super(DecodeLayer, self).__init__()
        self.default_boxes = anchor
        self.num_class = NUM_CLASSES + 1

    @tf.function
    def get_offset(self, predict_loc, box_pred):
        # 计算边框的中心点与宽高 (x,y,w,h)
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

    # @tf.function
    def nms(self, boxes, classes, scores):
        # mask = box_scores >= CONFIDENCE_THRESHOLD
        mask = tf.greater_equal(scores, CONFIDENCE_THRESHOLD)
        boxes = tf.boolean_mask(boxes, mask=mask, axis=0)
        scores = tf.boolean_mask(scores, mask=mask, axis=0)
        classes = tf.boolean_mask(classes, mask=mask, axis=0)

        box_list = []
        score_list = []
        class_list = []
        for i in range(1, self.num_class):
            mask = tf.equal(classes, i)
            box_of_class = tf.boolean_mask(boxes, mask)
            score_of_class = tf.boolean_mask(scores, mask)
            selected_indices = tf.image.non_max_suppression(boxes=box_of_class,
                                                            scores=score_of_class,
                                                            max_output_size=MAX_BOXES_PER_IMAGE,
                                                            iou_threshold=NMS_IOU_THRESHOLD)
            selected_boxes = tf.gather(box_of_class, selected_indices)
            selected_scores = tf.gather(score_of_class, selected_indices)
            classes = tf.ones_like(selected_scores, dtype=tf.dtypes.int32) * i
            box_list.append(selected_boxes)
            score_list.append(selected_scores)
            class_list.append(classes)
        box_tensor = tf.concat(values=box_list, axis=0)
        score_tensor = tf.concat(values=score_list, axis=0)
        class_tensor = tf.concat(values=class_list, axis=0)
        return box_tensor, score_tensor, class_tensor

    def call(self, logits, **kwargs):
        detection_boxes = []
        detection_classes = []
        detection_scores = []
        detection_num = []
        # y_true = tf.boolean_mask(logits, tf.not_equal(logits[..., -1], 0))
        # y_pred = tf.boolean_mask(logits, tf.not_equal(y_true[..., -1], 0))
        for logit in logits:
            predict_cls = logit[..., 4:]
            predict_loc = logit[..., :4]

            predict_loc = self.get_offset(predict_loc, self.default_boxes)

            scores = tf.reduce_max(predict_cls, axis=-1)
            classes = tf.argmax(predict_cls, axis=-1)

            mask = tf.not_equal(classes, 0)
            boxes = tf.boolean_mask(predict_loc, mask=mask, axis=0)
            score = tf.boolean_mask(scores, mask=mask, axis=0)
            cls = tf.boolean_mask(classes, mask=mask, axis=0)
            # selected_indices = tf.image.non_max_suppression(box, score, max_output_size=MAX_BOXES_PER_IMAGE,
            #                                                 iou_threshold=NMS_IOU_THRESHOLD)
            # box_tensor, score_tensor, class_tensor = self.nms(boxes, cls, score)
            # print("selected_indices:", selected_indices)
            detection_boxes.append(boxes)
            detection_classes.append(cls)
            detection_scores.append(score)
            detection_num.append(len(boxes))
        result = {"detection_boxes": detection_boxes, "detection_classes": detection_classes,
                  "detection_scores": detection_scores, "detection_num": detection_num}
        return result
