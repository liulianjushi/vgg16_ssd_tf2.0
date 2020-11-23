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
    def sigmoid_focal_loss(y_true, y_pred, alpha, gamma, from_logits=False):
        '''
        loss = -alpha*((1-p)^gamma)*log(p)
        :param y_true:
        :param y_pred:
        :param alpha:
        :param gamma:
        :return:
        '''
        ce = tf.keras.backend.binary_crossentropy(target=y_true, output=y_pred, from_logits=from_logits)
        if from_logits:
            pred_prob = tf.nn.softmax(y_pred)
        else:
            pred_prob = y_pred
        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.math.pow((1.0 - p_t), gamma)
        focal_loss = tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
        return tf.reduce_mean(focal_loss, axis=-1)

    @staticmethod
    def smooth_l1_loss(y_true, y_pred):
        absolute_value = tf.math.abs(y_true - y_pred)
        mask_boolean = tf.math.greater_equal(x=absolute_value, y=1.0)
        mask_float32 = tf.cast(x=mask_boolean, dtype=tf.float32)
        smooth_l1_loss = tf.reduce_sum((1.0 - mask_float32) * 0.5 * tf.math.square(absolute_value) + mask_float32 * (
                absolute_value - 0.5), axis=-1)
        return tf.reduce_mean(smooth_l1_loss, axis=-1)

    @staticmethod
    def __cover_background_boxes(true_boxes):
        symbol = true_boxes[..., -1]
        mask_symbol = tf.where(symbol < 0.5, 0.0, 1.0)
        mask_symbol = tf.expand_dims(input=mask_symbol, axis=-1)
        cover_boxes_tensor = tf.tile(input=mask_symbol, multiples=tf.constant([1, 1, 4], dtype=tf.dtypes.int32))
        return cover_boxes_tensor

    def __call__(self, y_true, y_pred):
        y_trues = tf.boolean_mask(y_true, tf.not_equal(y_true[..., -1], 0))
        y_preds = tf.boolean_mask(y_pred, tf.not_equal(y_true[..., -1], 0))

        true_class = tf.cast(y_trues[..., -1], dtype=tf.int32)
        pred_class = y_preds[..., 4:]
        true_class = tf.one_hot(indices=true_class, depth=self.num_classes, axis=-1)
        class_loss_value = self.sigmoid_focal_loss(y_true=true_class, y_pred=pred_class, alpha=self.alpha,
                                                   gamma=self.gamma)

        # cover_boxes = self.__cover_background_boxes(true_boxes=y_true)
        reg_loss_value = self.smooth_l1_loss(y_true=y_trues[..., :4], y_pred=y_preds[..., :4])
        loss = self.cls_loss_weight * class_loss_value + self.reg_loss_weight * reg_loss_value
        return loss, class_loss_value, reg_loss_value
