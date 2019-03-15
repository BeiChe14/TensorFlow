# -*- coding: utf-8 -*-
# @Time    : 2019/3/15 11:20
# @Author  : Chord
import tensorflow as tf;
from tensorflow.python.framework import graph_util;

v1 = tf.Variable(1, dtype=tf.float32, name="v1")
v2 = tf.Variable(2, dtype=tf.float32, name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    # 导出当前计算图的GraphDef部分，只需要这一部分就可以完成从输入层到输出层的计算过程
    graph_def = tf.get_default_graph().as_graph_def()

    # 将图中的变量及其取值转化为常量，同时将图中不必要的节点去掉。
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph_def, ['add']
    )
    # 将导出的模型存入文件
    with tf.gfile.GFile("C:\Documents\TF_resource\model\combined_model.pb", "wb") as f:
        f.write(output_graph_def.SerializeToString())

from tensorflow.python.platform import gfile;
with tf.Session() as sess:
    model_filename = "C:\Documents\TF_resource\model\combined_model.pb"
    # 读取保存的模型文件，并将文件解析成对应的GraphDef Protocol Buffer
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 将graph_def中保存的图加载到当前的图中。return_elements={"add:0"}给出了返回
    # 的张量名称。在保存的时间给出的是计算节点的名称，为“add”。加载时给出的是张
    # 量名，所以是“add:0”
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    print(sess.run(result))
