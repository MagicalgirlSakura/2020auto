import keras
from keras import datasets
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import numpy as np
import pickle as p
import load
from tensorflow import keras
import tensorflow as tf
import h5py

if __name__=="__main__":

    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    label_names = ['airplane', 'automobile', 'brid', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    y = tf.squeeze(y, axis=1)

    (x_new, y_new) = (x, y)

    for i in range(0, 100):
        #显示原图像
        plt.imshow(x[i])
        plt.figure(i)
        plt.show()
        xx_image = tf.image.random_flip_left_right(x[i])  #左右翻转
        xx_image = tf.image.random_flip_up_down(x[i]) #上下翻转
        xx_image = tf.image.random_hue(xx_image, max_delta=0.3) #调色度
        xx_image = tf.image.random_saturation(xx_image,lower=0.2,upper=1.8) #调饱和度
        xx_image = tf.image.random_brightness(xx_image, max_delta=0.7)  #调亮度
        xx_image = tf.image.random_contrast(xx_image, lower=0.2, upper=1.8)  #调对比度
        #显示随机变换后得到图像
        plt.imshow(xx_image)
        plt.figure(i + 1)  #防止图像覆盖
        plt.show()
        #将变换的图像插入原始训练数据集
        x_new = np.insert(x_new, 0, xx_image, axis=0)
        y_new = np.insert(y_new, 0, y[i], axis=0)


    np.savez("data_augment_new_cifar10", images=x_new, labels=y_new)

    new_data = np.load('data_augment_new_cifar10.npz')  #加载文件
    xx = new_data['images']
    yy = new_data['labels']

    model_names10 = ['CNN_with_dropout.h5','CNN_without_dropout.h5','lenet5_with_dropout.h5','lenet5_without_dropout.h5','random1_cifar10.h5','random2_cifar10.h5','ResNet_v1.h5','ResNet_v2.h5']
    #model_names100 = ["CNN_with_dropout.h5",'CNN_without_dropout.h5','lenet5_with_dropout.h5','lenet5_without_dropout.h5','random1_cifar100.h5','random2_cifar100.h5','ResNet_v1.h5','ResNet_v2.h5']
    model_dir_10 = "../model/cifar10/"
    #model_dir_100 = "../model/cifar100/"
    #model_dir_10 = "../model/cifar10/"

    scores = []
    for x in range(8):
        model = keras.models.load_model(model_dir_10 + model_names10[x])
        y_pred = model.predict(xx)
        count = 0
        for i in range(len(y_pred)):
            if (np.argmax(y_pred[i]) == yy[i]):
                count += 1
        score = count / len(y_pred)
        print(model_names10[x], '正确率为:%.2f%s' % (score * 100, '%'))
        scores.append(score)
    print(scores)



