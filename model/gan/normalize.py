import numpy as np
from sklearn.preprocessing import normalize
from math import log10, floor
import pandas as pd

def significant_number(x):
	return round(x, -floor(log10(abs(x))) )

def normalize(x):
	norm1 = x / np.linalg.norm(x)
	return norm1


data_file = '1600_embedding.txt'
data = np.loadtxt(data_file)

print(data.shape)

data_normed = []
for i in range(len(data)):
	data_normed.append(normalize(data[i]))
	print('check unit vector')
	print(i, " : ", np.linalg.norm(data_normed[i]))


matrix = pd.DataFrame(data = np.zeros([len(data_normed), len(data_normed)]), columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

for i in range(len(data_normed)):
	for j in range(len(data_normed)):
		matrix.iloc[i, j] = round(np.dot(data_normed[i], data_normed[j]), 2)

matrix.to_csv('1600_orthogonality_matrix.csv')

