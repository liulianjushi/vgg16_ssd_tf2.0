import tensorflow as tf

from configuration import NUM_CLASSES


class SSDLoss(object):
    def __init__(self, reg_loss_weight=0.5, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
        self.reg_loss_weight = reg_loss_weight
        self.cls_loss_weight = 1 - reg_loss_weight
        self.num_classes = NUM_CLASSES + 1

    @staticmethod
    def sigmoid_focal_loss(y_true, y_pred, alpha, gamma):
        sigmoid_loss = tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred, from_logits=True)
        pred_prob = tf.sigmoid(y_pred)
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_weight_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))
        sigmoid_focal_loss = modulating_factor * alpha_weight_factor * sigmoid_loss
        return sigmoid_focal_loss

    @staticmethod
    def smooth_l1_loss(y_true, y_pred):
        absolute_value = tf.math.abs(y_true - y_pred)
        mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
        mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
        smooth_l1_loss = (1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (
                absolute_value - 0.5)
        return smooth_l1_loss

    @staticmethod
    def __cover_background_boxes(true_boxes):
        symbol = true_boxes[..., -1]
        mask_symbol = tf.where(symbol < 0.5, 0.0, 1.0)
        mask_symbol = tf.expand_dims(input=mask_symbol, axis=-1)
        cover_boxes_tensor = tf.tile(input=mask_symbol, multiples=tf.constant([1, 1, 4], dtype=tf.dtypes.int32))
        return cover_boxes_tensor

    def __call__(self, y_true, y_pred):
        true_class = tf.cast(x=y_true[..., -1], dtype=tf.dtypes.int32)
        pred_class = y_pred[..., 4:]
        true_class = tf.one_hot(indices=true_class, depth=self.num_classes, axis=-1)
        class_loss_value = tf.math.reduce_sum(
            self.sigmoid_focal_loss(y_true=true_class, y_pred=pred_class, alpha=self.alpha, gamma=self.gamma))

        cover_boxes = self.__cover_background_boxes(true_boxes=y_true)
        # mask = tf.not_equal(y_true[..., -1], 0)
        # true_coord = tf.boolean_mask(y_true[..., :4], mask, axis=0)
        # pred_coord = tf.boolean_mask(y_pred[..., :4], mask, axis=0)
        reg_loss_value = tf.math.reduce_sum(
            self.smooth_l1_loss(y_true=y_true[..., :4], y_pred=y_pred[..., :4]))
        loss = self.cls_loss_weight * class_loss_value + self.reg_loss_weight * reg_loss_value
        return loss, class_loss_value, reg_loss_value
