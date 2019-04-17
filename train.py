#coding=utf-8
import numpy as np
import cv2
import glob
from sklearn.utils import shuffle
import tensorflow as tf
from skimage import io
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#Adding Seed so that random initialization is consistent

#def loadpic(file_name,file_pic):
#def load_lighthouse( ):
urls = np.loadtxt('Lighthouses.txt', dtype='U100')
if (os.path.exists("E:\\Github\\PycharmProjects\\DeepL\\dataSet\\Lighthouses")):
    shutil.rmtree("dataSet\\Lighthouses")
os.makedirs("dataSet\\Lighthouses")
# read the first image
for i in range(len(urls)):
    url = urls[i]
    img = io.imread(url)
    io.imshow(img)
    io.show()
    io.imsave('E:/Github/PycharmProjects/DeepL/dataSet/Lighthouses/' + repr(i) + '.png', img)
    #'D:/Workspace/Tensorflow/DeepL/dataSet/'

#def load_birdnests( ):
urls = np.loadtxt('Birdnests.txt', dtype='U100')
if (os.path.exists("E:\\Github\\PycharmProjects\\DeepL\\dataSet\\Birdnests")):
    shutil.rmtree("dataSet\\Birdnests")
os.makedirs("dataSet\\Birdnests")
# read the first image
for i in range(len(urls)):
    url = urls[i]
    img = io.imread(url)
    io.imshow(img)
    io.show()
    io.imsave('E:/Github/PycharmProjects/DeepL/dataSet/Birdnests/' + repr(i) + '.png', img)

#def load_honeycombs( ):
urls = np.loadtxt('Honeycombs.txt', dtype='U100')
if (os.path.exists("E:\\Github\\PycharmProjects\\DeepL\\dataSet\\Honeycombs")):
    shutil.rmtree("dataSet\\Honeycombs")
os.makedirs("dataSet\\Honeycombs")
# read the first image
for i in range(len(urls)):
    url = urls[i]
    img = io.imread(url)
    io.imshow(img)
    io.show()
    io.imsave('E:/Github/PycharmProjects/DeepL/dataSet/Honeycombs/' + repr(i) + '.png', img)


#######################################################################################################################
#######################################################################################################################
#Feature extraction



























#######################################################################################################################
#######################################################################################################################
#加载数据
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []  #class指三类图像
    print('Going to read training images')
    for fields in classes:   #分别遍历三类图像
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        #取到每个图像的绝对路径
        files = glob.glob(path)
        for fl in files:
            #读取图像、改变大小
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            #确保数据没问题
            image = image.astype(np.float32)
            #归一化
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            #狗对应的index是1
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    return images, labels, img_names, cls

class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

  #为了防止网络一直先看猫再看狗，打乱输入顺序
  images, labels, img_names, cls = load_train(train_path, image_size, classes)
  #shuffle改变输入图像的顺序
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

  if isinstance(validation_size, float):
    validation_size   = int(validation_size * images.shape[0])
  #测试集和训练集定义
  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]
  #实际构建一个类
  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets

#######################################################################################################################
#######################################################################################################################

#一次迭代的数据量
batch_size = 3

#Prepare input data 制定标签
classes = ['Birdnests', 'Honeycombs', 'Lighthouses']
num_classes = len(classes) #输出类别个数

# 20% of the data will automatically be used for validation
validation_size = 0.1    #100张有10%作为测试
img_size = 128            #图片的大小（基本都是正方形）
num_channels = 3
train_path = 'dataSet' #训练集的路径

data = read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

#######################################################################################################################
#######################################################################################################################

session = tf.Session()
# 固定大小
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

## labels两分类
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)  # 哪个值大就是什么

##Network graph params 卷积核设置
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 4096


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    layer = tf.nn.relu(layer)

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    # layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [batch_size img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


# 拉长全连接层
def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases

    layer = tf.nn.dropout(layer, keep_prob=0.7)

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
#Learning Rate
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session.run(tf.global_variables_initializer())


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss, i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1, i, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()

def train(num_iteration):
    global total_iterations

    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)

        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples / batch_size) == 0:
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples / batch_size))
            # 模型的保存
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss, i)
            saver.save(session, './model/machine-learning.ckpt', global_step=i)
            # meta.网络结构图
            # index.
            # DATA 权重参数

    total_iterations += num_iteration

train(num_iteration=108)