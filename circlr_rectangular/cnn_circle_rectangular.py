
import torch
from torchvision import models,transforms
from torch import optim,nn
from torch.autograd import Variable
from torch.utils import data
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split



class CircleRec(data.Dataset):
	def __init__(self,root,transform=None,train=False,test=False):
		self.train = train 
		self.test = test 
		self.transform = transform
		data = pd.read_csv(root)
		self.test_id=data['id']
		if self.train:
			samples,labels = data.iloc[:,1:-1],data['y']
			self.samples = samples.values.reshape(-1,1,40,40) / 255.
			self.labels = labels.values
		if self.test:
			self.samples = data.iloc[:,1:].values.reshape(-1,1,40,40) / 255.

	def __getitem__(self,index):
		sample = torch.Tensor(self.samples[index].astype(float))
		if self.train :
			label = self.labels[index]
			return sample,label
		if self.test :
			return sample 
	def __len__(self):
		return self.samples.shape[0]	
	def get_testid(self):
		return self.test_id

def get_data(root_train=None,root_test=None):
	train = CircleRec(root_train,train=True)
	test = CircleRec(root_test,test=True)
	test_id = test.get_testid()
	train_loader = data.DataLoader(train,batch_size=batch_size)
	test_loader = data.DataLoader(test,batch_size=batch_size)
	return train_loader,test_loader,test_id

class CNN(nn.Module):
	def __init__(self):
		super().__init__() #b,1,40,40
		self.layer1 = nn.Sequential(nn.Conv2d(1,8,3,1,1), #b,8,40
			nn.BatchNorm2d(8),nn.ReLU(True))
		self.layer2 = nn.Sequential(nn.Conv2d(8,16,3,1,1), #b,16,40,
			nn.BatchNorm2d(16),nn.ReLU(True),nn.MaxPool2d(2)) # b,16,20
		self.layer3 = nn.Sequential(nn.Conv2d(16,32,3,1,1), 
			nn.BatchNorm2d(32),nn.ReLU(True),nn.MaxPool2d(2)) #b,32,10
		self.layer4 = nn.Sequential(nn.Conv2d(32,64,3,1,1),
			nn.BatchNorm2d(64),nn.ReLU(True),nn.MaxPool2d(2))#b,64,5,5
		self.fc = nn.Sequential(nn.Linear(64*5*5,256),nn.ReLU(True),
			nn.Linear(256,16),nn.ReLU(True),nn.Dropout2d(),nn.Linear(16,2))

	def forward(self,x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x) 
		x = x.view(x.size(0),-1)
		x = self.fc(x)
		return x


def train_circle():
	model.train()
	train_acc,train_batch,correct,count,acc = [],[],0,0,0
	for epoch in range(n_epochs):
		for batch_idx,(img,label) in enumerate(train_loader,start=1):
			img,label = Variable(img),Variable(label)
			out = model(img)
			count += img.size(0)
			optimizer.zero_grad()
			loss = criterion(out,label)
			loss.backward()
			optimizer.step()

			_,pred = torch.max(out,1)
			correct += pred.eq(label).sum().item()
			acc = correct / count #pi:真实值 qi:预测概率值  #交叉熵 -sum(pi*logqi)
			train_batch.append(batch_idx) #pi为0 则[1,0] -1*log(-0.15)
			train_acc.append(acc) #pi所在索引若较大，则正确，否则计算损失
			print('Epoch={} [{:03d}]/[{}] | acc={:.5%} | loss={:.7f}'.format(
			epoch+1,batch_idx,len(train_loader),acc,loss.mean()))
	#torch.save(model.state_dict(),r'cirRec_Cnn.pkl')
	return model 

def test_criRec(model=None,test_id=None):
	model.eval()
	preds = [] 
	for imga in test_loader:
		img = Variable(imga)
		out = model(img)
		pred = torch.max(out,1)[1]
		prediction = pred.numpy().tolist()
		preds.extend(prediction)
	submission = pd.DataFrame(dict(id=test_id,y=preds))
	submission.to_csv(r'submission2.csv',index=False)
	
if __name__ == "__main__":

	batch_size,lr,n_epochs,eval_loss,eval_acc=32,8.4e-4,100,[],[]
	train_path,test_path = r'train.csv',r'test.csv'
	train_loader,test_loader,test_id = get_data(train_path,test_path)

	model = CNN()
	print(model)
	criterion = nn.CrossEntropyLoss() #SGD:momentum=0.9
	optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=3.4e-5)

	net = train_circle()
	test_criRec(model=net,test_id=test_id)
