import Initialization
import Globals
import numpy as np
import csv

if __name__ == '__main__':
	Initialization.initialization()
	data = Initialization.readInData('./data/data_train.csv')
	csvWriter = csv.writer(open('./log/itemRatingCount.csv','w',newline=''))
	for i in range(Globals.nItems):
		c = np.count_nonzero(data[:,i])
		csvWriter.writerow([c])
