
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 


def _show(k:int):
	show_pic = data.iloc[k,:].values.reshape(-1,40)
	plt.imshow(show_pic)
	plt.title('actual label=%d' % label[k])
	plt.show()

if __name__ == "__main__":
	data = pd.read_csv(r'train.csv')
	print(data.info())
	label = data['y']
	data = data.drop(['id','y'],axis=1)
	_show(3)