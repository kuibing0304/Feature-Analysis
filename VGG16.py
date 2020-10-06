#!/usr/bin/env python
# coding=utf-8

import time
import os
import sys
import gc
import logging
import numpy as np
import colorsys
# import png
import math
from functools import partial
from PIL import Image
from sklearn import preprocessing
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy.misc

# import utils

# logging.basicConfig(level=logging.DEBUG,
#                 format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='/home/s-20/python-homework/tudou/project/homework/python-test/Image/log/alexnet.log',
#                 filemode='w')
# logging.debug("This is debug message")

# TensorBoard可视化网络结构和参数
'''
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
flags.DEFINE_string('summaries_dir', '/tmp/AlexNet_logs', 'Summaries directory')
'''

'''把labels转换为one-hot形式'''
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def dense_to_one_hot(labels_dense, num_classes=10):
    logging.debug( "Convert class labels from scalars to one-hot vectors.")
    label_size = labels_dense.shape[0]

    # enc = preprocessing.OneHotEncoder(sparse=True, n_values=num_classes)
    enc = preprocessing.OneHotEncoder(n_values=num_classes)
    enc.fit(labels_dense)

    array = enc.transform(labels_dense).toarray()

    return array


# 对原始数据洗牌，跑不动
def shuffle_1(*arrs):
    start = time.time()
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))

    # logging.debug("==排序后的索引值====%s,%s"% (type(p), p))
    print("==排序后的索引值====%s,%s" % (type(p), p))
    data_shape = arrs[0].shape
    # logging.debug( "=====data_shape==========%s"% data_shape)
    print("=====data_shape==========%s" % data_shape)
    new_data = np.empty(data_shape, np.float)
    # logging.debug( "new_data占内存%s" % sys.getsizeof(new_data))
    print("new_data占内存%s" % sys.getsizeof(new_data))
    new_label = np.empty(arrs[1].shape, np.float)
    # 最蠢的方法
    data = arrs[0]
    label = arrs[1]
    for i in range(len(data)):
        tmp_data = data[p[i]]
        new_data[i] = tmp_data

        tmp_label = label[p[i]]
        new_label[i] = tmp_label
    end = time.time()
    # logging.debug( "shuffle花费的时间：%d" % ((end - start) / 1000))
    print("shuffle花费的时间：%d" % ((end - start) / 1000))
    return new_data, new_label


# 对原始数据洗牌，跑不动
def shuffle_2(*arrs):
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)


# 只获取洗牌后的索引
def shuffle_3(size):
    p = np.random.permutation(size)
    return p


'''NumPy的数组没有这种动态改变大小的功能，numpy.append()函数每次都会重新分配整个数组，并把原来的数组复制到新数组中
append效率太低。一次性把 数组 大小建好，再改改里面的数据即可，最后一步截取有效文件个数前size数据'''


def getImageMatrix(input_path='/raid/Wei/Codes/TestCode/PartitioningCNN/FeaturesAnalysis/VGG16/MINI_ImageNet/PartialImageNet/ZoomImg(1300)/'):
    # 获取目录下文件分类
    folderList = os.listdir(input_path)

    # 所有文件的个数
    size = len(sum([i[2] for i in os.walk(input_path)], []))
    # 存放data
    image_list = np.empty((size, 227, 227, 3), np.float)
    # logging.debug( "=====初始化image_list======",image_list.shape[0],image_list.shape

    # 存放labels
    labels_list = np.empty((0,10), np.float)
    # 要替换的data的索引
    index = 0
    # 遍历子目录，把类别转成int型===========遍历文件方式太复杂！！！！！！！！！待改!!!!!!!!!!!!!!
    for label, folder_name in enumerate(folderList, start=0):
        files = os.path.join(input_path, folder_name)
        # logging.debug( "目录的名字：%s" %  files)
        print("目录的名字：%s" % files)
        # 每个分类下有效文件的个数
        file_size = 0
        # 遍历目录，获取每个图像,变成227*227*3的向量
        for parent, dirnames, file_list in os.walk(files):
            file_size = len(file_list)
            print("====目录总文件的个数：%d" % file_size)
            for file in file_list:
                # 通过调用 array() 方法将图像转换成NumPy的数组对象
                image_path = os.path.join(parent, file)
                image = np.array(Image.open(image_path))
                # 判断图片的维数是否相同，过滤黑白的图片
                if image.shape == (227, 227, 3):
                    image_list[index] = image
                    index += 1
                    # image_list = np.append(image_list, image_list[index], axis = 0)
                else:
                    print("!!!!!!!!!!!!!!!!!格式不对删除图片!!!!!!!!!!!!!!!!!!!!!!!!!!!!%s" % image_path)
        # 获取label的one-hot矩阵
        labels = np.array([label] * file_size).reshape(-1, 1)
        # 目录下格式合格的文件不为空
        if labels.size:
            labels_one_hot = dense_to_one_hot(labels)
            labels_list = np.append(labels_list, labels_one_hot, axis=0)

    print("总文件个数：%d" % size)
    print("label的个数:%d" % labels_list.shape[0])
    print("image的个数:%d" % image_list.shape[0])
    print("image的占数据大小:%d" % sys.getsizeof(image_list))

    '''直接对原始数据洗牌，内存错误
    # 洗牌
    image_list_new, labels_list_new = shuffle(image_list, labels_list)
    # 分70%做train，30%做验证集算准确率
    train_size = int(size * 0.7)
    train_data = image_list_new[:train_size]
    train_label = labels_list_new[:train_size]
    validation_data = image_list_new[train_size:]
    validation_label = labels_list_new[train_size:]
    '''
    # 获取洗牌后的索引
    temp= len(labels_list)
    index_shuffle = shuffle_3(len(labels_list))
    # 分70%做train，30%做验证集算准确率
    # train_size = int(size * 0.7)
    train_size = int(size * 0.9)
    train_index = index_shuffle[:train_size]
    validation_index = index_shuffle[train_size:]
    # logging.debug( "训练的个数:%d" % len(train_index))
    # logging.debug( "验证数据的个数:%d" % len(validation_index))
    print("训练的个数:%d" % len(train_index))
    print("验证数据的个数:%d" % len(validation_index))
    return image_list, labels_list, train_index, validation_index


# 卷积层
def conv2d(name, input, w, b, stride, padding='SAME'):
    # 测试
    x = input.get_shape()[-1]

    x = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding=padding)

    x = tf.nn.bias_add(x, b)

    data_result = tf.nn.relu(x, name=name)
    # 输出参数
    # tf.histogram_summary(name + '/卷积层', data_result)
    gc.collect()
    return data_result


# 最大下采样
def max_pool(name, input, k, stride):
    return tf.nn.max_pool(input, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME', name=name)


# 归一化操作 ToDo 正则方式待修改
def norm(name, input, size=4):
    return tf.nn.lrn(input, size, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


# 定义整个网络 input：227*227*3
def VGG16(x, weights, biases, dropout):

    #Layer1
    conv11 = conv2d('conv11', x, weights['wc11'], biases['bc11'], stride=2)
    norm11 = norm('norm1', conv11, size=2)
    conv12 = conv2d('conv12', norm11, weights['wc12'], biases['bc12'], stride=1)
    pool11 = max_pool('pool11', conv12, k=3, stride=2)
    norm12 = norm('norm1', pool11, size=2)

    # Layer2
    conv21 = conv2d('conv21', norm12, weights['wc21'], biases['bc21'], stride=1)
    norm21 = norm('norm2', conv21, size=2)
    conv22 = conv2d('conv22', norm21, weights['wc22'], biases['bc22'], stride=1)
    pool21 = max_pool('pool21', conv22, k=3, stride=2)
    norm22 = norm('norm2', pool21, size=2)

    # Layer3
    conv31 = conv2d('conv31', norm22, weights['wc31'], biases['bc31'], stride=1)
    norm31 = norm('norm3', conv31, size=2)
    conv32 = conv2d('conv32', norm31, weights['wc32'], biases['bc32'], stride=1)
    pool31 = max_pool('pool31', conv32, k=3, stride=2)
    norm32 = norm('norm3', pool31, size=2)

    # Layer4
    conv41 = conv2d('conv41', norm32, weights['wc41'], biases['bc41'], stride=1)
    norm41 = norm('norm4', conv41, size=2)
    conv42= conv2d('conv42', norm41, weights['wc42'], biases['bc42'], stride=1)
    norm42= norm('norm4', conv42, size=2)
    conv43 = conv2d('conv43', norm42, weights['wc43'], biases['bc43'], stride=1)
    pool41 = max_pool('pool41', conv43, k=3, stride=2)
    norm43= norm('norm4', pool41, size=2)

    # Layer4
    conv51 = conv2d('conv51', norm43, weights['wc51'], biases['bc51'], stride=1)
    norm51 = norm('norm5', conv51, size=2)
    conv52 = conv2d('conv52', norm51, weights['wc52'], biases['bc52'], stride=1)
    norm52 = norm('norm5', conv52, size=2)
    conv53 = conv2d('conv53', norm52, weights['wc53'], biases['bc53'], stride=1)
    pool51 = max_pool('pool51', conv53, k=3, stride=2)
    # norm5 = norm('norm5', pool5, size=2)

    # 全连接层1
    fc1 = tf.reshape(pool51, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1, name='fc1')
    # Dropout
    drop1 = tf.nn.dropout(fc1, dropout)

    # 全连接层2
    fc2 = tf.add(tf.matmul(drop1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2, name='fc2')
    # Dropout
    drop2 = tf.nn.dropout(fc2, dropout)

    # out
    out = tf.add(tf.matmul(drop2, weights['out']), biases['out'])

    return out


if __name__ == '__main__':
    start_time = time.time()
    # 准备数据
    # train_data, train_label, validation_data, validation_label= getImageMatrix()
    image_list, labels_list, train_index, validation_index = getImageMatrix()
    end_time = time.time()
    # logging.debug( "准备数据的时间开销：%d" % (end_time - start_time))
    print("准备数据的时间开销：%d" % (end_time - start_time))

    # TODO 迁移数据，使用AlexNet

    # 训练参数
    learning_rate = 0.001
    training_iters =2
    batch_size =64
    display_step = 10

    # network 参数s
    n_input = [None, 227, 227, 3]
    n_classes = 10
    dropout =1.0
    Loss_CNN_imagenet=[]
    Loss_CNN_fabricImages=[]
    Accuracy_CNN_fabricImages=[]
    # tf graph input
    x = tf.placeholder(tf.float32, n_input, name='x')
    y = tf.placeholder(tf.float32, [None, n_classes], name='y')
    keep_prob = tf.placeholder(tf.float32)

    # 存储所有的参数
    weights = {
        'wc11': tf.Variable(tf.random_normal([3, 3, 3, 64])),
        'wc12': tf.Variable(tf.random_normal([3, 3, 64, 64])),
        'wc21': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'wc22': tf.Variable(tf.random_normal([3, 3, 128, 128])),
        'wc31': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'wc32': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        'wc41': tf.Variable(tf.random_normal([3, 3, 256, 512])),
        'wc42': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'wc43': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'wc51': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'wc52': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'wc53': tf.Variable(tf.random_normal([3, 3, 512, 512])),
        'wd1': tf.Variable(tf.random_normal([8192, 8192])),
        'wd2': tf.Variable(tf.random_normal([8192, 8192])),
        'out': tf.Variable(tf.random_normal([8192, 10]))
    }

    biases = {
        'bc11': tf.Variable(tf.random_normal([64])),
        'bc12': tf.Variable(tf.random_normal([64])),
        'bc21': tf.Variable(tf.random_normal([128])),
        'bc22': tf.Variable(tf.random_normal([128])),
        'bc31': tf.Variable(tf.random_normal([256])),
        'bc32': tf.Variable(tf.random_normal([256])),
        'bc41': tf.Variable(tf.random_normal([512])),
        'bc42': tf.Variable(tf.random_normal([512])),
        'bc43': tf.Variable(tf.random_normal([512])),
        'bc51': tf.Variable(tf.random_normal([512])),
        'bc52': tf.Variable(tf.random_normal([512])),
        'bc53': tf.Variable(tf.random_normal([512])),
        'bd1': tf.Variable(tf.random_normal([8192])),
        'bd2': tf.Variable(tf.random_normal([8192])),
        'out': tf.Variable(tf.random_normal([10]))
    }

    # 预测值
    pred = VGG16(x, weights, biases, keep_prob)

    # 定义损失函数和学习步骤

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # 初始化所有的共享变量
    init = tf.initialize_all_variables()

    '''以下的操作全是针对索引，而不是数据本身'''
    with tf.Session() as sess:
        sess.run(init)
        # 在traindata上做训练
        start_train_time = time.time()
        for i in range(training_iters):
            print("第%d轮:" % i)
            # print("对数据的索引再洗牌，获取新的索引列表")
            index_tmp = shuffle_3(len(train_index))
            train_index = np.array(train_index)
            train_index_new = train_index[index_tmp]

            data_size = len(train_index)
            start = 0
            while start < data_size:
                batch_index = train_index_new[start: start + batch_size]
                s_batch_time = time.time()
                batch_x = image_list[batch_index]
                batch_y = labels_list[batch_index]
                e_batch_time = time.time()
                # 喂数据
                s_batch_train_time = time.time()
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
                e_batch_train_time = time.time()
                cost_result=sess.run(cost,feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

                # print('cost:%3.10f' %cost_result)
                if (start) % 10000 == 0:
                    accuracy_result=sess.run(accuracy,feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    print ("training accuracy %1.3f"%sess.run(accuracy,feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}))
                    # Loss_CNN_imagenet.append(cost_result)
                    Accuracy_CNN_fabricImages.append(accuracy_result)
                start += batch_size
            #每轮释放一次内存
            gc.collect()
        end_train_time = time.time()
        print("Optimization Finished!")
        print("The Training times:",(end_train_time-start_train_time))
        gc.collect()
        # 在testdata上做训练
        # 测试准确度

        # np.savetxt(r'/home/scw4750/Wei/codes/ImportantCore/VisualMemoryCNN/TwoStreamCNN2/result/PartialImageNet/AccuracyThree/Accuracy_CNN_fabricImages.csv', Accuracy_CNN_fabricImages, delimiter=',')
        # np.savetxt(r'/home/scw4750/Wei/codes/ImportantCore/VisualMemoryCNN/TwoStreamCNN2/result/PartialImageNet/LossThree/Loss_CNN_imagenet.csv',Loss_CNN_imagenet, delimiter=',')
        # np.savetxt(r'/home/scw4750/Wei/codes/ImportantCore/VisualMemoryCNN/TwoStreamCNN2/result/CNN//ImageNet_loss/Loss_CNN_imagenet.csv',Loss_CNN_imagenet, delimiter=',')  # need modify 4,5 palces
        # plt.xlim(0, 350)
        # plt.ylim(0, 1)
        # plt.xlabel("steps")
        # plt.ylabel("loss")
        # plt.plot(Loss_CNN_imagenet)
        # plt.show()
        # print("shouw OK!")


        #get the First layer feature maps
        validation_data = image_list[validation_index]
        validation_label = labels_list[validation_index]

        # Get the feature maps conv1
        Second_feature1 = sess.graph.get_operation_by_name("conv11").outputs[0]
        Second_feature_out = sess.run(Second_feature1,feed_dict={x: validation_data[0:1], y: validation_label[0:1],keep_prob: 1.0})
        Second_feature_out = Second_feature_out[0]
        Second_feature_out = np.array(Second_feature_out)

        #Saving the feature maps one by one
        # for i in range(0, 64):
        #     feature_map="/raid/Wei/Codes/TestCode/PartitioningCNN/FeaturesAnalysis/VGG16/MINI_ImageNet/Feature_results/Feature"+str(i)+".jpg"
        #     scipy.misc.imsave(feature_map, Second_feature_out[i])

        # Save the feature maps
        np.save(file='feature_data_Conv1.npy', arr=Second_feature_out)

        #Clustering the feature maps  Using T-sne
        # First_feature_out=First_feature_out[0]
        # First_feature_out=First_feature_out.reshape((5776, 96))
        # model=TSNE(learning_rate=100)
        # transformed=model.fit_transform(First_feature_out)
        # x_axis=transformed[:, 0]
        # y_axis=transformed[:, 1]
        # plt.scatter(x_axis,y_axis, c=y_axis)
        # plt.show()




        #Testing the model
        Start_test_time=time.time()
        validation_data = image_list[validation_index]
        validation_label = labels_list[validation_index]
        # result_accracy = sess.run(accuracy, feed_dict={x: validation_data[0:50], y: validation_label[0:50], keep_prob: 1.})
        # result_cost = sess.run(cost, feed_dict={x: validation_data[0:50], y: validation_label[0:50], keep_prob: 1.})
        # result_pred = sess.run(pred, feed_dict={x: validation_data[0:50], keep_prob: 1.})
        # result_y = sess.run(y, feed_dict={y: validation_label[0:50]})
        # np.savetxt(r'/home/scw4750/Wei/codes/ImportantCore/AlexNetBradatz/image-recognition-TensorFlow-Alexnet-master/csvfiles/resultpred.csv',result_pred, delimiter=',')
        # np.savetxt(r'/home/scw4750/Wei/codes/ImportantCore/AlexNetBradatz/image-recognition-TensorFlow-Alexnet-master/csvfiles/result_y.csv',result_y, delimiter=',')
        # print('Accuracy on Test-dataset:%1.5f'%sess.run(accuracy,feed_dict={x:validation_data[0:497],y:validation_label[0:497],keep_prob:1.0}))
        # End_test_time=time.time()
        # print("The test time:", (End_test_time-Start_test_time))
