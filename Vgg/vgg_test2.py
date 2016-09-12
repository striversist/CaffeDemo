import caffe
import numpy as np

deploy = '/home/aaron/Downloads/vgg/dictnet_vgg_deploy.prototxt'
model = '/home/aaron/Downloads/vgg/dictnet_vgg.caffemodel'
labels_file = '/home/aaron/Downloads/vgg/dictnet_vgg_labels.txt'

img_file01 = '../images/ocr/373_coley_14845.jpg'
img_file02 = '../images/ocr/img_20.jpg'

caffe.set_device(0)
caffe.set_mode_gpu()

labels = np.loadtxt(labels_file, str, delimiter='\t')
net = caffe.Classifier(deploy, model, image_dims=(32, 100), raw_scale=255)
image = caffe.io.load_image(img_file01, False)

prediction = net.predict([image], False)
print '--------------------'
print 'image: ', img_file01
print 'prediction shape: ', prediction[0].shape
print 'predicted class:  ', prediction[0].argmax()
print 'predicted label: ', labels[prediction[0].argmax()]


word_rect = (95, 295, 546, 466)  # coordinates of rect (x1,y1, x2,y2) for word 'food' in picture img_20.jpg
image = caffe.io.load_image(img_file02, False)
image = image[word_rect[1]:word_rect[3], word_rect[0]:word_rect[2], :]
prediction = net.predict([image], False)
print '--------------------'
print 'image: ', img_file02
print 'prediction shape: ', prediction[0].shape
print 'predicted class:  ', prediction[0].argmax()
print 'predicted label: ', labels[prediction[0].argmax()]
