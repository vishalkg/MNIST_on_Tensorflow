import csv
import os
import numpy
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

file_path = os.path.dirname(os.path.realpath(__file__))
files = os.listdir(file_path)

class DataSet(object):

    def __init__(self,
                images,
                labels,
                fake_data=False,
                one_hot=False,
                dtype=dtypes.float32,
                reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
            self._images = images
            self._labels = labels
            self._epochs_completed = 0
            self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]

def get_file_name(file_name):
    return file_path+files[files.index(file_name)]

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def read_files(fake_data=False,
               one_hot=True,
               dtype=dtypes.float32,
               reshape=False,
               validation_size=3000,
               test_size=2000):
    #construct train data from csv
    #index = files.index("train.csv")
    x = 28
    y = 28
    labels = []
    images = []
    raw_data = []
    with open(file_path+"/train.csv", "r") as train_file:
        next(train_file)
        for data in csv.reader(train_file):
            raw_data.append(data)
            #labels.append(data[0])
            #images.append(data[1:])
    raw_data = numpy.array(raw_data,dtype=numpy.uint8)
    numpy.random.shuffle(raw_data)
    print(raw_data.shape)
    #train_labels = numpy.array(labels, dtype=numpy.uint8)
    #train_images = numpy.array(images, dtype=numpy.uint8)
    train_labels = raw_data[:,0]
    train_images = raw_data[:,1:]
    
    #images = []
    #labels = []
    test_set = []
    with open(file_path+"/test.csv", "r") as test_file:
        next(test_file)
        for data in csv.reader(test_file):
            test_set.append(data)

    test_set = numpy.array(test_set, dtype=numpy.uint8)
    test_set = test_set.reshape(len(test_set), x*y)

    #validation_images = train_images[:validation_size]
    #validation_labels = train_labels[:validation_size]
    test_labels = train_labels[:test_size]
    test_images = train_images[:test_size]
    train_images = train_images[test_size:]
    train_labels = train_labels[test_size:]

    validation_images = test_images
    validation_labels = test_labels

    #train_images = train_images.reshape(len(train_images), x*y)
    #test_images = test_images.reshape(len(test_images), x*y)

    train_labels = dense_to_one_hot(train_labels, 10)
    test_labels = dense_to_one_hot(test_labels, 10)

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    
    validation = DataSet(validation_images,
                        validation_labels,
                        dtype=dtype,
                        reshape=reshape)
    
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    return (base.Datasets(train=train, validation=validation, test=test), test_set)

if __name__ == '__main__':
    read_files()
    print ("completed")
