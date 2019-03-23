# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 20:58
# @Author  : Chord
import os;

import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data

# 加载mnist_inference.py中定义的常量和方法
import mnist_inference

# 配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "C:\Documents\TF_resource\model\\"
MODEL_NAME = "model.ckpt"

def train(mnist):
    # 调整输入数据格式，输入一个四维矩阵
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.IMAGE_SIZE,
        mnist_inference.NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input'
    )

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 直接使用mnist_inference,py中定义的前向传播过程
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variable_average_op = variable_average.apply(
        tf.trainable_variables()
    )

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )

    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
                          .minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 在训练过程中不再测试模型在验证数据上的表现，验证和测试的
        # 过程将会由一个独立的程序来完成
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.IMAGE_SIZE,
                                          mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x:reshaped_xs, y_:ys})
            # 每1000轮保存一次模型
            if(i % 1000 == 0):
                # 输出当前的训练情况，这里只输出了模型在当前训练batch上的损失
                # 函数大小。通过损失函数大小可以大概了解训练的情况。在验证数据
                # 集上的正确率信息会有一个单独的程序来生成
                print("After %d training step(s), loss on training "
                      "batch is %g." % (step, loss_value))
                # 保存当前的模型。注意这里给出了global_step参数，这样可以让每
                # 个被保存模型的文件名末尾都加上训练的轮数
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step
                )

def main(argv=None):
    mnist = input_data.read_data_sets("C:\Documents\TF_resource\MNIST_data", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()