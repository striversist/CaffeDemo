# Make sure that caffe is on the python path:
caffe_root = '../../../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np
import time

deploy = '/home/aaron/projects/caffe/examples/chinese_doc/6330/6330_chinese_91.4/vgg_chinese_6330_deploy.prototxt'
model = '/home/aaron/projects/caffe/examples/chinese_doc/6330/6330_chinese_91.4/ocr_chinese_6330_iter_8000.caffemodel'
labels_file = '/home/aaron/projects/caffe/examples/chinese_doc/6330/6330_chinese_91.4/label.txt'

img_file01 = '../images/ocr/chi_tian.png'
img_file02 = '../images/ocr/01souti.png'
img_file03 = '../images/ocr/01souti_bin.png'


caffe.set_device(0)
caffe.set_mode_cpu()

labels = np.loadtxt(labels_file, str, delimiter='\t')
net = caffe.Classifier(deploy, model, image_dims=(36, 36), raw_scale=255)

pred_start = time.time()
image = caffe.io.load_image(img_file01, False)
prediction = net.predict([image], False)
print '--------------------'
print 'image: ', img_file01
print 'prediction shape: ', prediction[0].shape
print 'predicted class:  ', prediction[0].argmax()
print 'predicted label: ', labels[prediction[0].argmax()]


word_rect = (629, 21, 629+36, 21+33)  # coordinates of rect (x1,y1, x2,y2) for word 'food' in picture img_20.jpg
image = caffe.io.load_image(img_file02, False)
print image.shape
image = image[word_rect[1]:word_rect[3], word_rect[0]:word_rect[2], :]
print image.shape
prediction = net.predict([image], False)
print '--------------------'
print 'image: ', img_file02
print 'prediction shape: ', prediction[0].shape
print 'predicted class:  ', prediction[0].argmax()
print 'predicted label: ', labels[prediction[0].argmax()]

word_rect = (521, 4, 521+38, 4+35)  # coordinates of rect (x1,y1, x2,y2) for word 'food' in picture img_20.jpg
image = caffe.io.load_image(img_file03, False)
print image.shape
image = image[word_rect[1]:word_rect[3], word_rect[0]:word_rect[2], :]
print image.shape
prediction = net.predict([image], False)
print '--------------------'
print 'image: ', img_file03
print 'prediction shape: ', prediction[0].shape
print 'predicted class:  ', prediction[0].argmax()
print 'predicted label: ', labels[prediction[0].argmax()]
print 'total used: {:.3f}'.format(time.time() - pred_start)
