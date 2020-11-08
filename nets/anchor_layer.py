import tensorflow as tf

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, ASPECT_RATIOS, ANCHOR_SIZES


class PriorBox(tf.keras.layers.Layer):
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None, offset=0.5, clip=True):
        super(PriorBox, self).__init__()
        self.img_width, self.img_height = img_size
        self.offset = offset
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = aspect_ratios
        self.clip = clip

    def call(self, input_shape, **kwargs):
        layer_width = input_shape[0]
        layer_height = input_shape[1]
        x, y = tf.meshgrid(tf.range(layer_width), tf.range(layer_height))
        x = (tf.cast(x, dtype=tf.float32) + self.offset) * tf.cast((self.img_width / layer_height), tf.float32)
        y = (tf.cast(y, dtype=tf.float32) + self.offset) * tf.cast((self.img_height / layer_width), tf.float32)

        w = tf.concat(([self.min_size, tf.sqrt(self.min_size * self.max_size)],
                       self.min_size * tf.sqrt(self.aspect_ratios)), axis=0)

        h = tf.concat(([self.min_size, tf.sqrt(self.min_size * self.max_size)],
                       self.min_size / tf.sqrt(self.aspect_ratios)), axis=0)

        wh = tf.stack((w, h), axis=1)
        center_xy = tf.stack((x, y), axis=2)
        xywh_list = [tf.concat([tf.tile(tf.reshape(xy, (1, -1)), [wh.shape[0], 1]), wh], axis=1) for xy in
                     tf.reshape(center_xy, (-1, 2))]
        prior_boxes = tf.concat(
            [[[xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2]] for
             xywhs in xywh_list for xywh in xywhs], axis=0)

        # 归一化，并处理超出边界的边框
        prior_boxes = tf.clip_by_value(
            prior_boxes / [self.img_width, self.img_height, self.img_width, self.img_height], 0.0, 1.0)

        return prior_boxes


class Anchor(tf.keras.layers.Layer):
    def __init__(self):
        super(Anchor, self).__init__()
        self.offset = 0.5
        self.image_width = IMAGE_HEIGHT
        self.image_height = IMAGE_WIDTH
        self.anchor_sizes = ANCHOR_SIZES
        self.aspect_ratios = ASPECT_RATIOS
        self.conv4_3_priorbox = PriorBox(img_size=(300, 300), min_size=30., max_size=60.,
                                         aspect_ratios=[2.0, 0.5])
        self.conv7_priorbox = PriorBox(img_size=(300, 300), min_size=60., max_size=111.,
                                       aspect_ratios=[2.0, 0.5, 3.0, 1.0 / 3.0])
        self.conv8_2_priorbox = PriorBox(img_size=(300, 300), min_size=111., max_size=162.,
                                         aspect_ratios=[2.0, 0.5, 3.0, 1.0 / 3.0])
        self.conv9_2_priorbox = PriorBox(img_size=(300, 300), min_size=162., max_size=213.,
                                         aspect_ratios=[2.0, 0.5, 3.0, 1.0 / 3.0])
        self.conv10_2_priorbox = PriorBox(img_size=(300, 300), min_size=213., max_size=264.,
                                          aspect_ratios=[2.0, 0.5])
        self.conv11_2_priorbox = PriorBox(img_size=(300, 300), min_size=264., max_size=315.,
                                          aspect_ratios=[2.0, 0.5])

    def call(self, inputs, **kwargs):
        conv4_3_priorbox = self.conv4_3_priorbox(inputs[0])
        conv7_priorbox = self.conv7_priorbox(inputs[1])
        conv8_2_priorbox = self.conv8_2_priorbox(inputs[2])
        conv9_2_priorbox = self.conv9_2_priorbox(inputs[3])
        conv10_2_priorbox = self.conv10_2_priorbox(inputs[4])
        conv11_2_priorbox = self.conv11_2_priorbox(inputs[5])

        priorbox = tf.concat([conv4_3_priorbox, conv7_priorbox, conv8_2_priorbox, conv9_2_priorbox, conv10_2_priorbox,
                              conv11_2_priorbox], axis=0)

        return priorbox
