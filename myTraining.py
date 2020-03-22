import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

def data_split(data,ratio):
  np.random.seed(42)
  shuffled = np.random.permutation(len(data))
  test_set_size = int(len(data)*ratio)
  test_indices = shuffled[:test_set_size]
  train_indices = shuffled[test_set_size:]
  return(data.iloc[train_indices],data.iloc[test_indices])


if __name__ == '__main__':
	data = pd.read_csv('data.csv')
	
	train,test = data_split(data,0.2)

	x_train = train[['Fever','BodyPain',	'Age','RunnyNoise','DifficultBreadth']].to_numpy()
	x_test = test[['Fever','BodyPain',	'Age','RunnyNoise','DifficultBreadth']].to_numpy()
	y_train = train[['InfectionProbability']].to_numpy().reshape(1252,-1)
	y_test = test[['InfectionProbability']].to_numpy().reshape(313,-1)

	clf = LogisticRegression()
	clf.fit(x_train,y_train)

	file = open('model.pkl','wb')
	pickle.dump(clf,file)
	file.close()