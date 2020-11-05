import tensorflow as tf

from configuration import IMAGE_HEIGHT, IMAGE_WIDTH, ASPECT_RATIOS, ANCHOR_SIZES, ANCHOR_STEPS


class Anchor(tf.keras.layers.Layer):
    def __init__(self):
        super(Anchor, self).__init__()
        self.offset = 0.5
        self.image_width = IMAGE_HEIGHT
        self.image_height = IMAGE_WIDTH
        self.anchor_steps = ANCHOR_STEPS
        self.anchor_sizes = ANCHOR_SIZES
        self.aspect_ratios = ASPECT_RATIOS

    def call(self, inputs, **kwargs):
        feature_map_boxes = []
        for i, size in enumerate(inputs):
            y, x = tf.meshgrid(tf.range(size[0]), tf.range(size[1]))
            y = (tf.cast(y, dtype=tf.float32) + self.offset) * self.anchor_steps[i]
            x = (tf.cast(x, dtype=tf.float32) + self.offset) * self.anchor_steps[i]

            w = tf.concat(([self.anchor_sizes[i][0], tf.sqrt(self.anchor_sizes[i][0] * self.anchor_sizes[i][1])],
                           self.anchor_sizes[i][0] * tf.sqrt(self.aspect_ratios[i][1:])), axis=0)

            h = tf.concat(([self.anchor_sizes[i][0], tf.sqrt(self.anchor_sizes[i][0] * self.anchor_sizes[i][1])],
                           self.anchor_sizes[i][0] / tf.sqrt(self.aspect_ratios[i][1:])), axis=0)

            wh = tf.stack((w, h), axis=1)
            center_xy = tf.stack((x, y), axis=2)
            xywh_list = [tf.concat([tf.tile(tf.reshape(xy, (1, -1)), [wh.shape[0], 1]), wh], axis=1) for xy in
                         tf.reshape(center_xy, (-1, 2))]
            default_boxes = tf.concat(
                [[[xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[0] + xywh[2] / 2, xywh[1] + xywh[3] / 2]] for
                 xywhs in xywh_list for xywh in xywhs], axis=0)

            # 归一化，并处理超出边界的边框
            default_boxes = tf.clip_by_value(
                default_boxes / [self.image_width, self.image_height, self.image_width, self.image_height], 0.0, 1.0)

            feature_map_boxes.append(default_boxes)
        return tf.concat(feature_map_boxes, axis=0)


if __name__ == '__main__':
    anchor = Anchor()
    # feature_size = tf.constant()
    x = anchor([(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)])
    print(x.shape)

    # for y in x.numpy():
    #     print(y*300)
