# coding=utf-8

import sys
pycaffe='/home/aaron/projects/caffe/build/install/python'
sys.path.insert(0, pycaffe)
import caffe
import numpy as np


deploy = '/home/aaron/Downloads/vgg/dictnet_vgg_deploy.prototxt'  # deploy文件
caffe_model = '/home/aaron/Downloads/vgg/dictnet_vgg.caffemodel'  # 训练好的 caffemodel
img01 = '/home/aaron/share/image01.jpg'  # 随机找的一张待测图片
img02 = '/home/aaron/share/image02.jpg'  # 随机找的一张待测图片
imgs = [img01, img02]
labels_filename = '/home/aaron/Downloads/vgg/dictnet_vgg_labels.txt'  # 类别名称文件，将数字标签转换回类别名称

net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network

# 图片预处理设置
print '===== net.data.shape: ', net.blobs['data'].data.shape
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
# transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 255)  # 缩放到【0，255】之间
# transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

for img in imgs:
    print '===== load image: ', img
    im = caffe.io.load_image(img, False)  # 加载图片
    print '===== im.shape: ', im.shape
    net.blobs['data'].data[...] = transformer.preprocess('data', im)  # 执行上面设置的图片预处理操作，并将图片载入到blob中

    # 执行测试
    print '===== start forward'
    out = net.forward()
    print '===== forward finish'

    labels = np.loadtxt(labels_filename, str, delimiter='\t')  # 读取类别名称文件
    prob = net.blobs['prob'].data[0]  # 取出最后一层（Softmax）属于某个类别的概率值，并打印
    max_index = prob.argmax()
    print 'the most prob index{}, class is {}: '.format(max_index, labels[max_index])
    # sort top five predictions from softmax output
    top_inds = prob.flatten().argsort()[::-1][:5]  # reverse sort and take five largest items
    print 'probabilities and labels:'
    for ind in top_inds:
        print '===== index={}, value={}, class={}'.format(ind, prob.flatten()[ind], labels[ind])
