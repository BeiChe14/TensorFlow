# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 10:22
# @Author  : Chord

import tensorflow as tf;

# MNIST数据集相关的常数
INPUT_NODE = 784        # 输入层节点数，总像素数
OUTPUT_NODE = 10        # 输出层节点数，总类别数

# 配置神经网络参数
LAYER1_NODE = 500       # 隐藏层节点数，使用单隐藏层作为样例

BATCH_SIZE = 100        # 一个训练batch中的训练数据个数。数字越小越接近随机梯度下降，
                        # 数字越大越接近梯度下降
LEARNING_RATE_BASE = 0.8        # 基础学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率

def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope("layer1", reuse=reuse):
        # 根据传进来的reuse来判断是创建新变量还是使用已经创建好的。
        # 在第一次构造网络时需要创建新变量，以后每次调用这个函数都
        # 直接使用reuse=True，就不需要每次传递变量了
        weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似地定义第二层神经网络变量和前向传播结果
    with tf.variable_scope("layer2", reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    # 返回最后的前向传播结果
    return layer2

x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
y = inference(x)
