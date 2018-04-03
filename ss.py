import tensorflow as tf
slim = tf.contrib.slim
from matplotlib import pyplot as plt
import os

# from datasets.tfrecord_reader import tfrecord_reader
# dataset_dir = '/home/dan/prj/datasets/mnist/'
# dataset = tfrecord_reader.get_split('mnist', 'train', dataset_dir)
# with tf.Graph().as_default(): 
#     dataset_dir = '/home/dan/prj/datasets/mnist'
#     dataset = tfrecord_reader.get_split('mnist', 'train', dataset_dir)
    
#     data_provider = slim.dataset_data_provider.DatasetDataProvider(
#         dataset, common_queue_capacity=32, common_queue_min=1)
#     image, label = data_provider.get(['image', 'label'])
    
#     with tf.Session() as sess:    
#         with slim.queues.QueueRunners(sess):
#             for i in range(1):
#                 np_image, np_label = sess.run([image, label])
#                 height, width, _ = np_image.shape
#                 class_name = name = dataset.labels_to_names[np_label]
                
#                 plt.figure()
#                 plt.imshow(np_image[:,:,0], cmap='gray')
#                 #plt.imsave('/home/dan/'+str(i)+'.png',np_image[:,:,0], cmap='')
#                 #print(np_image[:,:,0])
#                 plt.title('%s, %d x %d' % (name, height, width))
#                 plt.axis('off')
#                 plt.show()


def test(tensor):
    return tensor, [2*tensor, 3*tensor], [4*tensor, 5*tensor]


tensor = tf.constant([1, 2, 3, 4])
with tf.Session() as sess:
    hi, hello, good = test(tensor)
    print(sess.run(hi))
    print(sess.run(hello))
    print(sess.run(good))
