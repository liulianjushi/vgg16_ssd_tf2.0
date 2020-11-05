import tensorflow as tf

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')
# if gpus:
#     try:
#         # 设置GPU显存占用为按需分配
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # 异常处理
#         print(e)
from configuration import NMS_IOU_THRESHOLD, CONFIDENCE_THRESHOLD
from nets.anchor_layer import Anchor
from nets.decode_layer import DecodeLayer
from nets.input_layer import InputLayer


class L2norm(tf.keras.layers.Layer):
    def __init__(self, gamma_init=20, trainable=True):
        super(L2norm, self).__init__()
        self.trainable = trainable
        self.gamma_init = gamma_init

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gamma = self.add_weight('gamma', shape=[channels, ], trainable=self.trainable) * self.gamma_init

    def call(self, inputs, **kwargs):
        l2_norm = tf.nn.l2_normalize(inputs, axis=-1, epsilon=1e-12)  # 只对每个像素点在channels上做归一化
        return l2_norm * self.gamma


class SsdLayer(tf.keras.Model):
    def __init__(self, num_classes=2):
        super(SsdLayer, self).__init__()

        self.num_classes = num_classes + 1
        self.conv1_1 = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv1_1")
        self.conv1_2 = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv1_2")

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same", name="pool1")
        self.conv2_1 = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2_1")
        self.conv2_2 = tf.keras.layers.Conv2D(128, 3, padding="same", name="conv2_2")

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same", name="pool2")
        self.conv3_1 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv3_1")
        self.conv3_2 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv3_2")
        self.conv3_3 = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv3_3")

        self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same", name="pool3")
        self.conv4_1 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv4_1")
        self.conv4_2 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv4_2")
        self.conv4_3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv4_3")
        self.l2norm = L2norm()
        self.predict_loc_1 = tf.keras.layers.Conv2D(4 * 4, 3, padding="same", name="predict_loc_1")
        self.predict_loc_reshape_1 = tf.keras.layers.Reshape([38 * 38 * 4, 4])
        self.predict_cls_1 = tf.keras.layers.Conv2D(4 * self.num_classes, 3, padding="same", name="predict_cls_1")
        self.predict_cls_reshape_1 = tf.keras.layers.Reshape([38 * 38 * 4, self.num_classes])

        self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding="same", name="pool4")
        self.conv5_1 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv5_1")
        self.conv5_2 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv5_2")
        self.conv5_3 = tf.keras.layers.Conv2D(512, 3, padding="same", activation="relu", name="conv5_3")

        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding="same", name="pool5")
        self.conv6 = tf.keras.layers.Conv2D(1024, 3, dilation_rate=6, activation="relu", padding="same", name="conv6")

        self.conv7 = tf.keras.layers.Conv2D(1024, 1, padding="same", activation="relu", name="conv7")

        self.predict_loc_2 = tf.keras.layers.Conv2D(6 * 4, 3, padding="same", name="predict_loc_2")
        self.predict_loc_reshape_2 = tf.keras.layers.Reshape([19 * 19 * 6, 4])
        self.predict_cls_2 = tf.keras.layers.Conv2D(6 * self.num_classes, 3, padding="same", name="predict_cls_2")
        self.predict_cls_reshape_2 = tf.keras.layers.Reshape([19 * 19 * 6, self.num_classes])

        self.conv8_1 = tf.keras.layers.Conv2D(256, 1, padding="same", activation="relu", name="conv8_1")
        self.conv8_2 = tf.keras.layers.Conv2D(512, 3, 2, padding="valid", activation="relu", name="conv8_2")

        self.predict_loc_3 = tf.keras.layers.Conv2D(6 * 4, 3, padding="same", name="predict_loc_3")
        self.predict_loc_reshape_3 = tf.keras.layers.Reshape([10 * 10 * 6, 4])
        self.predict_cls_3 = tf.keras.layers.Conv2D(6 * self.num_classes, 3, padding="same", name="predict_cls_3")
        self.predict_cls_reshape_3 = tf.keras.layers.Reshape([10 * 10 * 6, self.num_classes])

        self.conv9_1 = tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu", name="conv9_1")
        self.conv9_2 = tf.keras.layers.Conv2D(256, 3, 2, padding="valid", activation="relu", name="conv9_2")

        self.predict_loc_4 = tf.keras.layers.Conv2D(6 * 4, 3, padding="same", name="predict_loc_4")
        self.predict_loc_reshape_4 = tf.keras.layers.Reshape([5 * 5 * 6, 4])
        self.predict_cls_4 = tf.keras.layers.Conv2D(6 * self.num_classes, 3, padding="same", name="predict_cls_4")
        self.predict_cls_reshape_4 = tf.keras.layers.Reshape([5 * 5 * 6, self.num_classes])

        self.conv10_1 = tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu", name="conv10_1")
        self.conv10_2 = tf.keras.layers.Conv2D(256, 3, padding="valid", activation="relu", name="conv10_2")

        self.predict_loc_5 = tf.keras.layers.Conv2D(4 * 4, 3, padding="same", name="predict_loc_5")
        self.predict_loc_reshape_5 = tf.keras.layers.Reshape([3 * 3 * 4, 4])
        self.predict_cls_5 = tf.keras.layers.Conv2D(4 * self.num_classes, 3, padding="same", name="predict_cls_5")
        self.predict_cls_reshape_5 = tf.keras.layers.Reshape([3 * 3 * 4, self.num_classes])

        self.conv11_1 = tf.keras.layers.Conv2D(128, 1, padding="same", activation="relu", name="conv11_1")
        self.conv11_2 = tf.keras.layers.Conv2D(256, 3, padding="valid", activation="relu", name="conv11_2")

        self.predict_loc_6 = tf.keras.layers.Conv2D(4 * 4, 3, padding="same", name="predict_loc_6")
        self.predict_loc_reshape_6 = tf.keras.layers.Reshape([1 * 1 * 4, 4])
        self.predict_cls_6 = tf.keras.layers.Conv2D(4 * self.num_classes, 3, padding="same", name="predict_cls_6")
        self.predict_cls_reshape_6 = tf.keras.layers.Reshape([1 * 1 * 4, self.num_classes])
        self.decode_layer = DecodeLayer()

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1_1(inputs)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)

        l2norm = self.l2norm(x)
        predict_loc_1 = self.predict_loc_1(l2norm)
        predict_loc_1 = self.predict_loc_reshape_1(predict_loc_1)
        predict_cls_1 = self.predict_cls_1(l2norm)
        predict_cls_1 = self.predict_cls_reshape_1(predict_cls_1)
        # predict_cls_1 = tf.nn.softmax(predict_cls_1, axis=-1)

        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        predict_loc_2 = self.predict_loc_2(x)
        predict_loc_2 = self.predict_loc_reshape_2(predict_loc_2)
        predict_cls_2 = self.predict_cls_2(x)
        predict_cls_2 = self.predict_cls_reshape_2(predict_cls_2)
        # predict_cls_2 = tf.nn.softmax(predict_cls_2, axis=-1)

        x = self.conv8_1(x)
        x = self.conv8_2(tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]]))
        predict_loc_3 = self.predict_loc_3(x)
        predict_loc_3 = self.predict_loc_reshape_3(predict_loc_3)
        predict_cls_3 = self.predict_cls_3(x)
        predict_cls_3 = self.predict_cls_reshape_3(predict_cls_3)
        # predict_cls_3 = tf.nn.softmax(predict_cls_3, axis=-1)

        x = self.conv9_1(x)
        x = self.conv9_2(tf.pad(x, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]]))
        predict_loc_4 = self.predict_loc_4(x)
        predict_loc_4 = self.predict_loc_reshape_4(predict_loc_4)
        predict_cls_4 = self.predict_cls_4(x)
        predict_cls_4 = self.predict_cls_reshape_4(predict_cls_4)
        # predict_cls_4 = tf.nn.softmax(predict_cls_4, axis=-1)

        x = self.conv10_1(x)
        x = self.conv10_2(x)
        predict_loc_5 = self.predict_loc_5(x)
        predict_loc_5 = self.predict_loc_reshape_5(predict_loc_5)
        predict_cls_5 = self.predict_cls_5(x)
        predict_cls_5 = self.predict_cls_reshape_5(predict_cls_5)
        # predict_cls_5 = tf.nn.softmax(predict_cls_5, axis=-1)

        x = self.conv11_1(x)
        x = self.conv11_2(x)
        predict_loc_6 = self.predict_loc_6(x)
        predict_loc_6 = self.predict_loc_reshape_6(predict_loc_6)
        predict_cls_6 = self.predict_cls_6(x)
        predict_cls_6 = self.predict_cls_reshape_6(predict_cls_6)
        # predict_cls_6 = tf.nn.softmax(predict_cls_6, axis=-1)

        predict_cls = tf.concat(
            [predict_cls_1, predict_cls_2, predict_cls_3, predict_cls_4, predict_cls_5, predict_cls_6], 1)
        predict_loc = tf.concat(
            [predict_loc_1, predict_loc_2, predict_loc_3, predict_loc_4, predict_loc_5, predict_loc_6], 1)
        logit = tf.concat([predict_loc, predict_cls], 2)

        # ############# 解码 #####################
        if not training:
            return self.decode_layer(logit)
        return logit


if __name__ == '__main__':

    input = InputLayer(batch_size=4)
    dataset = input("data/val.record")
    ssd = SsdLayer(num_classes=20)
    ssd.build(input_shape=(None, 300, 300, 3))
    for step, (images, labels) in enumerate(dataset):
        output_dict = ssd(images["raw"])
        print(output_dict)
