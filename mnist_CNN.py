import numpy as np
import time
from datetime import timedelta
import math
import data_set
import tensorflow as tf

data, test = data_set.read_files()
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
data.test.cls = np.argmax(data.test.labels, axis=1)
print(set(data.test.cls))

input("Press enter to continue .. ")
# Convolutional Layer 1.
filter_size1 = 5          
num_filters1 = 32         

# Convolutional Layer 2.
filter_size2 = 5          
num_filters2 = 64         

# Convolutional Layer 3.
filter_size3 = 2          
num_filters3 = 128         

# Convolutional Layer 4.
filter_size4 = 2          
num_filters4 = 256         

# Fully-connected layers.
fc_size1 = 512
fc_size2 = 256

img_size = 28
feat_d = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

images = data.test.images[0:9]
cls_true = data.test.cls[0:9]

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, feat_d], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                                            num_input_channels=num_channels,
                                            filter_size=filter_size1,
                                            num_filters=num_filters1,
                                            use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                                            num_input_channels=num_filters1,
                                            filter_size=filter_size2,
                                            num_filters=num_filters2,
                                            use_pooling=False)

layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                                            num_input_channels=num_filters2,
                                            filter_size=filter_size3,
                                            num_filters=num_filters3,
                                            use_pooling=True)

layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                                            num_input_channels=num_filters3,
                                            filter_size=filter_size4,
                                            num_filters=num_filters4,
                                            use_pooling=False)

layer_flat, num_features = flatten_layer(layer_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size1,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size1,
                         num_outputs=fc_size2,
                         use_relu=True)

layer_fc3 = new_fc_layer(input=layer_fc2,
                         num_inputs=fc_size2,
                         num_outputs=num_classes,
                         use_relu=False)

y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, dimension=1)

l_rate = 1e-3
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(cost)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
'''
if not exists 'sess.chkp.meta':
	session.run(tf.global_variables_initializer())
else:
	restore_saver =tf.train.import_meta_graph('sess.chkp.meta')
	restore_saver.restore(session,tf.train.latest_checkpoint('./'))
'''
session.run(tf.global_variables_initializer())
batch_size = 64
total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    global batch_pointer
    global l_rate
    global max_iter

    batch_pointer = 0
    start_time = time.time()
    #data4graph = []
    for i in range(total_iterations,total_iterations + num_iterations):

        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch,y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            #feed_dict_validate = {x: data.test.images,y_true: data.test.labels}
            #acc = session.run(accuracy, feed_dict=feed_dict_validate)
            #data4graph.append((i,acc))
            acc = get_validation_accuracy()
            msg = "Optimization Iteration: {0:>6}, Validation Accuracy: {1:>6.1%}"
            print(msg.format(i+1, acc))

    total_iterations += num_iterations

    end_time = time.time()
    #f = open('data4graph.txt','w')
    #for i in range(len(data4graph)):
    #    f.write(str(data4graph[i])+'\n')
    #f.close()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    #_saver = tf.train.Saver()
    #_saver.save(session,"sess.chkp")

# Split the test-set into smaller batches of this size.
test_batch_size = 250
def get_validation_accuracy(show_example_errors=False,show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    
    
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    #print("- Validation-set:\t{}".format(len(data.validation.labels)))
    #msg = "Final Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    #print(msg.format(acc, correct_sum, num_test))
    return acc

def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
    num_test = len(data.test.images)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
    i = 0
    while i < num_test:
        j = min(i + test_batch_size, num_test)
        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: images,y_true: labels}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j
    
    
    cls_true = data.test.cls
    correct = (cls_true == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum) / num_test
    print("- Validation-set:\t{}".format(len(data.validation.labels)))
    msg = "Final Accuracy on Validation-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))
    #session.close()
'''
def make_submission():
    test_csv = open('test.csv','r').readlines()[1:]
    num_test = len(test_csv)
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    test_data = np.zeros((len(test_csv),784))
    for i in range(len(test_csv)):
        test_data[i] = [float(pixel) for pixel in test_csv[i].split(',')]

    f = open('submission11.csv','w')
    f.write('imageid,label\n')
    i = 0
    while i<len(test_csv):
        j = min(i+test_batch_size,len(test_csv))
        images = test[i:j,:]
        feed_dict = {x: images}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        for i in range(i,j):
            f.write(str(i+1)+','+str(cls_pred[i])+'\n')
        i = j
    f.close()
'''
def main():
	max_iter = 10000
	optimize(num_iterations=max_iter)
	print_test_accuracy()
	#make_submission()
	
if __name__=="__main__":
	main()

