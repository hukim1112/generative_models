
from implementations import infogan_various_figures, infogan_various_figures_tmp, infogan_various_figures_gray, infogan_mnist, megan_mnist, megan_various_figures
from datasets.data_generator import various_figures, various_figures_gray
from datasets.tfrecord_creator import converter
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path')
# parser.add_argument('--dataset_path')
# args = parser.parse_args()
# checkpoint_path = args.checkpoint_path
# dataset_path = args.dataset_path
batch_size = 128
checkpoint_path = '/home/dan/prj/weights/ex_mnist_1_infogan'
#dataset_path = '/home/dan/prj/datasets/various_figures_64x64'
#dataset_path = '/home/dan/prj/datasets/various_figures_64x64_no_angle_tri'
dataset_path = '/home/dan/prj/datasets/mnist'
#dataset_path = '/home/dan/prj/datasets/various_figures_64x64_gray'
result_path = '/home/dan/prj/results/ex_mnist_1_infogan'


#1. make dataset and tfrecord conversion
# various_figures_gray.run(dataset_path)
# datasetname = 'figures'
# _NUM_VALIDATION = 3000
# _NUM_SHARD = 5

# tfrecord = converter.tf_converter(datasetname, dataset_path, _NUM_VALIDATION, _NUM_SHARD)
# tfrecord.run()

#2. training and storing visual results at every 500 iterations
infogan_mnist.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result



#3. implementation

#infogan_various_figures.impl(checkpoint_path, dataset_path, batch_size)