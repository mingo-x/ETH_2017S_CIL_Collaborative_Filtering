import numpy as np
import csv

# read in training data
trainingData = np.empty((1000,1000))
csvReader = csv.reader(open('../data/'))