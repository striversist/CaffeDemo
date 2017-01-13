caffe_root = '../../../caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np


# demo
deploy = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(deploy, model, caffe.TEST)
for param in net.params:
    print '{} weights are {} dimensional and biases are {} dimensional.'.format(param, net.params[param][0].data.shape,
                                                                                net.params[param][1].data.shape)

deploy2 = caffe_root + 'models/bvlc_reference_caffenet/deploy_1001.prototxt'
net2 = caffe.Net(deploy2, caffe.TEST)
for param in net2.params:
    print '{} weights are {} dimensional and biases are {} dimensional.'.format(param, net2.params[param][0].data.shape,
                                                                                net2.params[param][1].data.shape)
extract_params = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']
for param in extract_params:
    net2.params[param][0].data[...] = net.params[param][0].data[...]
    net2.params[param][1].data[...] = net.params[param][1].data[...]

net2.save('output_layer_caffenet.caffemodel')
