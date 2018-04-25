
import tensorflow as tf
slim = tf.contrib.slim
from datasets.tfrecord_reader import tfrecord_reader
dataset_path = '/home/dan/prj/datasets/various_figures_64x64_gray'
dataset_name = 'figures'
split_name = 'train'
batch_size = 32

dataset = tfrecord_reader.get_split(dataset_name, split_name, dataset_path)
data_provider = slim.dataset_data_provider.DatasetDataProvider(
		            dataset, common_queue_capacity=4*batch_size, common_queue_min=batch_size)    
[image, label] = data_provider.get(['image', 'label'])
image = tf.image.rgb_to_grayscale(image)
image = (tf.to_float(image) - 128.0) / 128.0

with tf.Session() as sess:
	with slim.queues.QueueRunners(sess):
		image = sess.run(image)
		

print(image.shape)
print(image.max())