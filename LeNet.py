#包文件导入区
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf

from tensorflow.contrib.layers import flatten

def LeNet(x):  
    #输入维度是32*32*1
    # Arguments used for tf.truncated_normal, randomly defines variables for 
    #the weights and biases for each layer
    

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    
    return logits

#%%导入数据工作
#第一个参数显示数据集下载的位置
mnist = input_data.read_data_sets(".",reshape=False)

X_train = mnist.train.images
y_train = mnist.train.labels

X_validation = mnist.validation.images
y_validation = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

#测试一下调入的数据是否有异常
assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

#%%扩展图片过程
# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    
print("Updated Image Shape: {}".format(X_train[0].shape))

#%%查看一部分图片
'''
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
plt.show()
print(y_train[index])
'''
#%%重新排列数据库
X_train, y_train = shuffle(X_train, y_train)

#%%开始构建网络
EPOCHS = 10
BATCH_SIZE = 128

#输入和输出节点
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)

#连接训练节点
rate = 0.001

#%%网络构建过程
#logits = LeNet(x)
mu = 0
sigma = 0.1
    
# TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
#创建卷积层1
weight1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev = sigma))
bias1 = tf.Variable(tf.zeros(6))
conv_layer1 = tf.nn.conv2d(x, weight1, strides=[1, 1, 1, 1], padding='VALID')
conv_layer1 = tf.nn.bias_add(conv_layer1, bias1)
# TODO: Activation.Your choice of activation function.
OutPut1 = tf.nn.sigmoid(conv_layer1)
# TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
Pool1 = tf.nn.max_pool(OutPut1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
# TODO: Layer 2: Convolutional. Output = 10x10x16.
weight2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev = sigma))
bias2 = tf.Variable(tf.zeros(16))
conv_layer2 = tf.nn.conv2d(Pool1, weight2, strides=[1, 1, 1, 1], padding='VALID')
conv_layer2 = tf.nn.bias_add(conv_layer2, bias2)
# TODO: Activation.Your choice of activation function.
OutPut2 = tf.nn.sigmoid(conv_layer2)
# TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
Pool2 = tf.nn.max_pool(OutPut2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
# TODO: Flatten. Input = 5x5x16. Output = 400.
# Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. 
# The easiest way to do is by using tf.contrib.layers.flatten, which is already imported for you.
Flatten2 = flatten(Pool2)
# TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
weight3 = tf.Variable(tf.truncated_normal([400,120],mean = mu, stddev = sigma))
bias3 = tf.Variable(tf.zeros(120))
Mux3 = tf.matmul(Flatten2,weight3)
logits3 = tf.add(Mux3,bias3)
# TODO: Activation.Your choice of activation function.
OutPut3 = tf.nn.sigmoid(logits3)
# TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
weight4 = tf.Variable(tf.truncated_normal([120,84],mean = mu, stddev = sigma))
bias4 = tf.Variable(tf.zeros(84))
Mux4 = tf.matmul(OutPut3,weight4)
logits4 = tf.add(Mux4,bias4)
# TODO: Activation.Your choice of activation function.
OutPut4 = tf.nn.sigmoid(logits4)
# TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
weight5 = tf.Variable(tf.truncated_normal([84,10],mean = mu, stddev = sigma))
bias5 = tf.Variable(tf.zeros(10))
Mux5 = tf.matmul(OutPut4,weight5)
logits = tf.add(Mux5,bias5)

#%%生成代价函数
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

#%%生成优化器
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

#%%验证过程
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#%%执行网络训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")














