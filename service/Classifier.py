
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
	def __init__(self,ratio):
		fname = "data/dataset.csv"
		self.df = self.PreProcess(fname)
		self.ratio = ratio


	def GetFScore(self,j):
		if j == 0:
			return self.RunNN()
		if j == 1:
			return self.RFC()
		if j == 2:
			return self.DTree()
		return self.KNN()




	def TrainTestSplit(self):
		X = self.df.drop('Type', axis=1)  
		y = self.df['Type']
		return train_test_split(X, y, test_size = 1 - self.ratio)

	def RFC(self):
		clf = RandomForestClassifier(criterion='gini', random_state=20)
		return self.MinFScore(clf)


	def DTree(self):
		clf = tree.DecisionTreeClassifier()
		return self.MinFScore(clf)

	def KNN(self):
		clf = KNeighborsClassifier(n_neighbors=8)
		return self.MinFScore(clf)


	def F1_Score(self,y_test,y_pred):
		r = classification_report(y_test,y_pred)
		r = r.split('      ')
		r = r[len(r)-2]
		return float(r)

	def RunNN(self):
		clf = MLPClassifier(hidden_layer_sizes=(50, 50,50,50,50,50,50,50,10), max_iter=100000)
		return self.MinFScore(clf)

	def MinFScore(self,clf):
		mscore = 1.0
		for i in range(15):
			self.X_train, self.X_test, self.y_train, self.y_test = self.TrainTestSplit()
			clf.fit(self.X_train, self.y_train)
			y_pred = clf.predict(self.X_test)
			fscore = self.F1_Score(self.y_test,y_pred)
			if fscore < mscore:
				mscore = fscore
		return mscore

	def RunSVM(self):
		svclassifier = SVC(kernel='linear') 
		svclassifier.fit(self.X_train, self.y_train)
		y_pred = svclassifier.predict(self.X_test) 
		return self.F1_Score(self.y_test,y_pred)

	def PreProcess(self,fname):
		df = None
		if not os.path.exists('n.csv'):
			df = pd.read_csv(fname)
			df.fillna(-1, inplace=True)
			del df['URL']
			catColList = df.columns
			valColList = ['URL', 'URL_LENGTH', 'NUMBER_SPECIAL_CHARACTERS','Type','CONTENT_LENGTH']
			for col in catColList:
				if col not in valColList:
					df[col+'1'] = df[col].values
					df = self.ReplceCatByID(df,col)
			df = self.ToggleValues(df,'Type')
			df.to_csv('n.csv', sep=',', encoding='utf-8')
		else:
			df = pd.read_csv('n.csv')
		return df

	def ToggleValues(self,df,colName):
		for index, row in df.iterrows():
			df.at[index,colName] = 0 if row[colName] == 1 else  1
		return df

	def ReplceCatByID(self,df,colName):
		cats = df[colName].unique()
		dcats,conVal = {},{}
		c = 0
		for cat in cats:
			dcats[cat] = c
			conVal[c,0] = 0
			conVal[c,1] = 0
			c += 1


		for index, row in df.iterrows():
			df.at[index,colName] = dcats[row[colName]] if row[colName] in dcats else -1
			if row[colName] in dcats:
				idx = dcats[row[colName]]
				cl = row['Type']
				conVal[idx,cl] = conVal[idx,cl] + 1 

		for index, row in df.iterrows():
			v0,v1 = conVal[row[colName],0],conVal[row[colName],1]
			df.at[index,colName] = v0
			df.at[index,colName + '1'] = v0 + v1
		return df

