
from implementations import infogan_various_figures, infogan_various_figures_tmp
from datasets.data_generator import various_figures
from datasets.tfrecord_creator import converter
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoint_path')
# parser.add_argument('--dataset_path')
# args = parser.parse_args()
# checkpoint_path = args.checkpoint_path
# dataset_path = args.dataset_path
batch_size = 128
checkpoint_path = '/home/dan/prj/weights/infogan64x64/9th_ex'
#dataset_path = '/home/dan/prj/datasets/various_figures_64x64'
#dataset_path = '/home/dan/prj/datasets/various_figures_64x64_no_angle_tri'
dataset_path = '/home/dan/prj/datasets/mnist'
result_path = 'results/various_figures/64x64_9th'


#1. make dataset and tfrecord conversion
#various_figures.run(dataset_path)
# datasetname = 'figures'
# dataset_dir = '/home/dan/prj/datasets/various_figures'
# _NUM_VALIDATION = 3000
# _NUM_SHARD = 5

# tfrecord = converter.tf_converter(datasetname, dataset_dir, _NUM_VALIDATION, _NUM_SHARD)
# tfrecord.run()

#2. training and storing visual results at every 500 iterations
checkpoint_path = '/home/dan/prj/weights/infogan64x64/5th_ex'
result_path = 'results/various_figures/64x64_5th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64'
infogan_various_figures.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result

checkpoint_path = '/home/dan/prj/weights/infogan64x64/6th_ex'
result_path = 'results/various_figures/64x64_6th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64_no_angle_tri'
infogan_various_figures.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result

checkpoint_path = '/home/dan/prj/weights/infogan64x64/7th_ex'
result_path = 'results/various_figures/64x64_7th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64'
infogan_various_figures_tmp.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result

checkpoint_path = '/home/dan/prj/weights/infogan64x64/8th_ex'
result_path = 'results/various_figures/64x64_8th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64_no_angle_tri'
infogan_various_figures_tmp.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result


checkpoint_path = '/home/dan/prj/weights/infogan64x64/5th_ex'
result_path = 'results/various_figures_double_exp/64x64_5th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64'
infogan_various_figures.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result

checkpoint_path = '/home/dan/prj/weights/infogan64x64/6th_ex'
result_path = 'results/various_figures_double_exp/64x64_6th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64_no_angle_tri'
infogan_various_figures.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result

checkpoint_path = '/home/dan/prj/weights/infogan64x64/7th_ex'
result_path = 'results/various_figures_double_exp/64x64_7th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64'
infogan_various_figures_tmp.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result

checkpoint_path = '/home/dan/prj/weights/infogan64x64/8th_ex'
result_path = 'results/various_figures_double_exp/64x64_8th'
dataset_path = '/home/dan/prj/datasets/various_figures_64x64_no_angle_tri'
infogan_various_figures_tmp.train(checkpoint_path, dataset_path, batch_size, result_path) # training and show result


#3. implementation

#infogan_various_figures.impl(checkpoint_path, dataset_path, batch_size)