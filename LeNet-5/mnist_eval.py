# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 20:58
# @Author  : Chord
import time;
import tensorflow as tf;
import numpy as np;
from tensorflow.examples.tutorials.mnist import input_data;

import mnist_train, mnist_inference

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 5
BATCH_SIZE = 100

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.IMAGE_SIZE,
            mnist_inference.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input'
        )

        xs, ys = mnist.validation.next_batch(mnist_train.BATCH_SIZE)
        # 类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
        reshaped_xs = np.reshape(xs, (mnist_train.BATCH_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.IMAGE_SIZE,
                                      mnist_inference.NUM_CHANNELS))
        validate_feed = {
            x: reshaped_xs,
            y_: ys
        }


        # 直接通过调用封装好的函数来计算前向传播的结果。因为测试时不关注正则化损失的值
        # 所以这里用于计算正则化损失的函数被设置成None
        y = mnist_inference.inference(x, False, None)

        # 使用前向传播的结果计算正确率。如果需要对未知的样例进行分类，那么使用
        # tf.argmax(y,1)就可以得到输入样例的预测类别了
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均
        # 的函数来获取平均值了。这样就可以完全共用mnist_inference.py中定义的前向传播过
        # 程了
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY
        )
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # 每个EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化
        while (True):
            with tf.Session() as sess:
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到
                # 目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(
                    mnist_train.MODEL_SAVE_PATH
                )
                if (ckpt and ckpt.model_checkpoint_path):
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = ckpt.model_checkpoint_path\
                                      .split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation "
                          "avvuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECS)
    pass

def main(atgv=None):
    mnist = input_data.read_data_sets("C:\Documents\TF_resource\MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()