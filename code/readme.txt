[Introduction]
In this project, we implemented a collaborative filtering system which exploits both the memory-based method and the model-based methods. 14 predictors, including:

1-6.	6 basic predictors that use simple statistics (Basic.py), 
7.	the user-based neighbourhood-based method (UserBased.py),
8.	the k-means predictor (KMeans.py), 
9.	the SVD with dimension reduction (PlainSVD.py), 
10.	the regularized SVD (RSVDFull.py), 
11.	the biased regularized SVD (RSVDFull2.py), 
12-13.	the kernel ridge regression based on regularized SVD and on biased regularized SVD (KRR.py),
14.	and the linear weighted model (LinearModel.py)

were tuned to their best performance and combined. Ridge Regression took the predictions of the above models and the two-way interactions between some of them as features and the observed ratings as targets and fitted on a validation set. Thus, we obtained an ensemble of different methods. We generated two splits of the training set and the validation set, and ran Ridge Regression on each split. The final predictor of our system was the average of the two regression results.

[Required Setting]
Python3
sklearn
numpy

[Instructions]
In order to reproduce the prediction result, please follow the instructions. 

1. Structure the files and folders in the following way:
	./
		data/
			data_train.csv
			sampleSubmission.csv
		code/
			(put the code (.py files) here)
		log/

2. Run the following commands to train the models: (The training of some models could take hours.)
	# 6 basic predictors
	python3 code/Basic.py
	python3 code/Basic.py -d=1
	# userd-base model
	python3 code/UserBased.py
	python3 code/UserBased.py -d=1
	# k-means
	python3 code/KMeans.py
	python3 code/KMeans.py -d=1
	python3 code/KMeans.py -p=c
	python3 code/KMeans.py -p=c -d=1
	# SVD with dimension reduction
	python3 code/PlainSVD.py -k=12
	python3 code/PlainSVD.py -k=12 -d=1
	# regularized SVD
	python3 code/RSVDFull.py -k=32 -l=0.1 -l2=0.1
	python3 code/RSVDFull.py -k=32 -l=0.1 -l2=0.1 -d=1
	# biased regularized SVD
	python3 code/RSVDFull2.py -k=32 -l=0.1 -l2=0.1
	python3 code/RSVDFull2.py -k=32 -l=0.1 -l2=0.1 -d=1
	# kernel ridge regression
	python3 code/KRR.py -k=32
	python3 code/KRR.py -k=32 -d=1
	python3 code/KRR.py -i=2 -k=32
	python3 code/KRR.py -i=2 -k=32 -d=1
	# linear weighted model
	python3 code/LinearModel.py
	python3 code/LinearModel.py -d=1

3. Run the following commands to combine the models:
	python3 code/Ensemble.py -p=tr
	python3 code/Ensemble.py -p=tr -d=1
	python3 code/Ensemble.py -p=a -d=tr

4. The final prediction is stored in the "data" folder (prediction.csv).