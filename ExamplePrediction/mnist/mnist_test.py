import caffe


model = "lenet.prototxt"
weigths = "lenet_iter_10000.caffemodel"

caffe.set_device(0)
caffe.set_mode_gpu()

net = caffe.Classifier(model, weigths, image_dims=(28, 28), raw_scale=256)
image = [caffe.io.load_image("test.bmp", color=False)]
prediction = net.predict(image, False)
print prediction
print '--------------------'
print 'prediction shape: ', prediction[0].shape
print 'predicted class:  ', prediction[0].argmax()

