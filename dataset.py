
import tensorflow as tf
slim = tf.contrib.slim
from dataset.tfrecord_creator import converter
datasetname = 'flowers'
dataset_dir = '/home/dan/prj/generative_models/flower_photos'
_NUM_VALIDATION = 400
_NUM_SHARD = 5

tfrecord = converter.tf_converter(datasetname, dataset_dir, _NUM_VALIDATION, _NUM_SHARD)
tfrecord.run()