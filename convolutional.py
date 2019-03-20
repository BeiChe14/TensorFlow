# -*- coding: utf-8 -*-
# @Time    : 2019/3/20 20:36
# @Author  : Chord

import tensorflow as tf;
# 通过tf.get_variable的方式创建卷积核的权重变量和偏置项变量。上面介绍了卷积层
# 的参数个数只和卷积核尺寸、个数以及当前层节点矩阵的深度油管，所以这里声明的参
# 数变量是一个四维矩阵，前面两个维度带包了卷积核的尺寸，第三个尺寸表示当前层的
# 深度，第四个维度表示卷积核的个数
filter_weight = tf.get_variable(
    'weights', [5, 5, 3, 16],
    initializer=tf.truncated_normal_initializer(0.1)
)
# 和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层
# 深度个不同的偏置项，或者每个卷积核有一个偏置项。
biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1)
)

# tf.nn.conv2d提供了一个非常方便的函数来实现卷积层的前向传播算法。这个函数的第
# 一个输入为当前层的节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个
# 节点矩阵，第一维对应一个输入batch。比如在输入层，input[0,:,:,:]表示第一张图片,
# input[1,:,:,:]表示第二张图片。tf.nn.conv2d第二个参数提供了卷积层的权重，第三
# 个参数为不同维度上的步长。虽然第三个参数提供的是一个长度为4的数组，但是第一维
# 和最后一维的数字要求一定是1。这是因为卷积层的步长只队矩阵的长和宽有效。最后一
# 个参数是填充（padding）方法，TensorFlow中听过SAME或是VALID两种选择。启动SAME
# 表示全0填充，VALID表示不添加
conv = tf.nn.conv2d(
    input, filter_weight, strides=[1, 1, 1, 1], padding='SAME'
)

# tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用
# 加法，因为矩阵上不同位置上的几点都需要加上那个同样的偏置项。
bias = tf.nn.bias_add(conv, biases)
# 将计算结果通过ReLU激活函数完成去线性化
actived_conv = tf.nn.relu(bias)

# tf.nn.max_pool实现了最大池化层的前向传播过程，它的参数和tf.nn.conv2d函数类似
# ksize提供了过滤器的尺寸，strides提供了步长信息，padding提供了是否使用全0填充
pool = tf.nn.max_pool(actived_conv, ksize=[1, 3, 3, 1],
                      strides=[1, 2, 2, 1], padding='SAME')