from flask import Flask,render_template,request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()
 
@app.route('/',methods=["GET","POST"])
def hello_world():
	if request.method == "POST":
		myDict=request.form
		fever=float(myDict['fever'])
		age=int(myDict['age'])
		pain=int(myDict['pain'])
		runnyNose=int(myDict['runnyNose'])
		breath=int(myDict['breath'])
		inputFeatures = [fever,pain,age,runnyNose,breath]
		infProb=clf.predict_proba([inputFeatures])[0][1]
		return render_template('show.html',inf=round(infProb*100))
	return render_template('index.html')

if __name__ == '__main__':
	app.run(host='192.168.0.108',port=5000)