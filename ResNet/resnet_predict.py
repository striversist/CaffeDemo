import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would like to classify.
MODEL_FILE = caffe_root + 'models/resnet/ResNet-50-deploy.prototxt'
PRETRAINED = caffe_root + 'models/resnet/ResNet-50-model.caffemodel'
IMAGE_FILE = '../images/38/n01675722_4197.JPEG'

caffe.set_mode_cpu()
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       mean=np.load('ResNet_mean.npy').mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(224, 224))
input_image = caffe.io.load_image(IMAGE_FILE)

# predict takes any number of images, and formats them for the Caffe net automatically
prediction = net.predict([input_image])
print 'prediction shape:', prediction[0].shape
print 'predicted class:', prediction[0].argmax()

