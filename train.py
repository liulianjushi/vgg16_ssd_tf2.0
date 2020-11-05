import math
import time

import numpy as np
import tensorflow as tf

from configuration import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS, EPOCHS, BATCH_SIZE, MAP_SIZE
from nets.anchor_layer import Anchor
from nets.decode_layer import DecodeLayer
from nets.encode_layer import EncodeLayer
from nets.loss_layer import SSDLoss
from nets import dataset
from nets.vgg_layer import SsdLayer
from utils.visualization_utils import plot_to_image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    try:
        # 设置GPU显存占用为按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # 异常处理
        print(e)

ssd = SsdLayer(num_classes=NUM_CLASSES)
ssd.build(input_shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
print(ssd.summary())

# loss
loss = SSDLoss()

# optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                             decay_steps=5000,
                                                             decay_rate=0.96)
optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

# metrics
loss_metric = tf.metrics.Mean()
cls_loss_metric = tf.metrics.Mean()
reg_loss_metric = tf.metrics.Mean()

valid_loss_metric = tf.metrics.Mean()
valid_cls_loss_metric = tf.metrics.Mean()
valid_reg_loss_metric = tf.metrics.Mean()

print("loading dataset...")

train_dataset = dataset.VOCDataSet("data/val.record", batch_size=BATCH_SIZE, epoch=EPOCHS).load_data()
num = len(list(train_dataset.as_numpy_iterator()))
print(f"train dataset num:{num}")
val_dataset = dataset.VOCDataSet("data/val.record", batch_size=4, epoch=1).load_data()
print("finish loading dataset.")

anchor = Anchor()
anchor_boxes = anchor(MAP_SIZE)
encode_layer = EncodeLayer(anchor_boxes)
decode_layer = DecodeLayer(anchor_boxes)
print("start training...")

ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, net=ssd)
manager = tf.train.CheckpointManager(ckpt, directory='./models', checkpoint_name='model.ckpt', max_to_keep=10)

# Set up logging.
stamp = time.strftime("%Y%m%d%H%M", time.localtime(int(time.time())))

train_summary_writer = tf.summary.create_file_writer(f'./tensorboard/{stamp}/train/')  # 实例化记录器
valid_summary_writer = tf.summary.create_file_writer(f'./tensorboard/{stamp}/valid/')  # 实例化记录器
tf.summary.trace_on(graph=True, profiler=True)  # 开启Trace（可选）


@tf.function
def train_step(batch_images, batch_labels):
    with tf.GradientTape() as tape:
        logit = ssd(batch_images, training=True)
        loss_value, cls_loss, reg_loss = loss(y_true=batch_labels, y_pred=logit)
        loss_metric.update_state(values=loss_value)
        cls_loss_metric.update_state(values=cls_loss)
        reg_loss_metric.update_state(values=reg_loss)
    gradients = tape.gradient(loss_value, ssd.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, ssd.trainable_variables))
    return logit


@tf.function
def valid_step(batch_images, batch_labels):
    logit = ssd(batch_images, training=True)
    loss_value, cls_loss, reg_loss = loss(y_true=batch_labels, y_pred=logit)
    valid_loss_metric.update_state(values=loss_value)
    valid_cls_loss_metric.update_state(values=cls_loss)
    valid_reg_loss_metric.update_state(values=reg_loss)
    return logit


ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

for index, (images, labels) in enumerate(train_dataset):
    batch_labels = encode_layer(labels)
    logit = train_step(images["raw"], batch_labels)
    output = decode_layer(logit)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 5 == 0:
        save_path = manager.save(checkpoint_number=ckpt.step)
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        # for i, (images, labels) in enumerate(val_dataset):
        #     batch_labels = encode_layer(labels)
        #     logit = valid_step(images["raw"], batch_labels)
        #     output = decode_layer(logit)
        #     with valid_summary_writer.as_default():  # 指定记录器
        #         tf.summary.scalar("valid/valid_loss_metric", valid_loss_metric.result(), step=index)  # 将当前损失函数的值写入记录器
        #         tf.summary.scalar("valid/valid_cls_loss_metric", valid_cls_loss_metric.result(),
        #                           step=index)  # 将当前损失函数的值写入记录器
        #         tf.summary.scalar("valid/valid_reg_loss_metric", valid_reg_loss_metric.result(),
        #                           step=index)  # 将当前损失函数的值写入记录器
        #         tf.summary.image("valid/10 valid data examples", plot_to_image(images, labels), max_outputs=10,
        #                          step=index)
        # print(
        #     f"step:{index}/{num},valid_loss_metric:{loss_metric.result()},valid_cls_loss_metric:{valid_cls_loss_metric.result()},valid_reg_loss_metric:{valid_reg_loss_metric.result()}")
        # valid_loss_metric.reset_states()
        # valid_cls_loss_metric.reset_states()
        # valid_reg_loss_metric.reset_states()
    with train_summary_writer.as_default():  # 指定记录器
        tf.summary.scalar("train/loss_metric", loss_metric.result(), step=index)  # 将当前损失函数的值写入记录器
        tf.summary.scalar("train/cls_loss_metric", cls_loss_metric.result(), step=index)  # 将当前损失函数的值写入记录器
        tf.summary.scalar("train/reg_loss_metric", reg_loss_metric.result(), step=index)  # 将当前损失函数的值写入记录器
        tf.summary.image("train/10 train data examples", plot_to_image(images, labels, output), max_outputs=10,
                         step=index)
    print(
        f"step: {index}/{num},loss_metric:{loss_metric.result()},cls_loss_metric:{cls_loss_metric.result()},reg_loss_metric:{reg_loss_metric.result()}")
    loss_metric.reset_states()
    cls_loss_metric.reset_states()
    reg_loss_metric.reset_states()
with train_summary_writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir="tensorboard/train")  # 保存Trace信息到文件（可选）
