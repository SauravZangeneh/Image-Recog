import numpy as np
from PIL import Image
from random import choice

def img_preprocess(i):
	my_pic = Image.open("/Users/saurav/Desktop/kagglecatsanddogs_3367a/PetImages/Dog/" + str(i)+".jpg",'r')
	try:
		np.pad(my_pic,((1,1), (1,1), (0,0)), mode='constant')
	except ValueError:
		my_pic.convert('RGB').save('/Users/saurav/Desktop/kagglecatsanddogs_3367a/PetImages/Dog/'+ str(i)+".jpg")
		print("converted")

def gen_test_cat(i):
	my_pic = np.array(Image.open("/Users/saurav/Desktop/kagglecatsanddogs_3367a/PetImages/Cat/" + str(i)+".jpg",'r'))
	M=np.pad(my_pic,((1,1), (1,1), (0,0)), mode='constant')
	return M

def gen_test_dog(i):
	my_pic = np.array(Image.open("/Users/saurav/Desktop/kagglecatsanddogs_3367a/PetImages/Dog/" + str(i)+".jpg",'r'))
	M=np.pad(my_pic,((1,1), (1,1), (0,0)), mode='constant')
	return M

def img_2_vec(X):
	M=np.swapaxes(X,0,2)
	k,c,r=np.shape(M)
	i_hor=np.tile(np.repeat(np.arange(3),3),3)
	i_ver=np.repeat(np.arange(c-2),r-2)
	i=i_hor.reshape(1,-1) + i_ver.reshape(-1,1)
	j_hor=np.tile(np.tile(np.arange(3),3),3)
	j_ver=np.tile(np.arange(r-2),c-2)
	j=j_ver.reshape(-1,1) + j_hor.reshape(1,-1)
	k=np.repeat(np.arange(3),9).reshape(1,-1)+np.repeat(0,(r-2)*(c-2)).reshape(-1,1)

	return M[k,i,j]

def batch(start,s):
	Lay_1in=[]
	out=[]
	functions=[gen_test_cat,gen_test_dog]
	for m in range(start,start+s):
		fns=choice(functions)
		if (fns==functions[0]):
			y=0
		else:
			y=1
		t=img_2_vec(fns(m))
		d=np.resize(t,(200000,27))
		Lay_1in.append(d)
		out.append(y)
	q=np.dstack(Lay_1in)
	return out,q

def conv(W,B,X):
	out=np.tensordot(X,W,axes=([1],[0])) + B 
	return(out)
def conv2(W,B,X):
	out=np.tensordot(X,W,axes=([2],[0])) + B
	return out
def Rel(X):
	val=np.maximum(X,0)
	delta=np.ones_like(X)
	delta[X<0]=0
	return val,delta

def activ(X,y1):
	yh=np.array(1/(1+np.exp(-X)))
	y=np.array(y1)
	loss=-y*np.log(yh) - (1-y)*np.log(1-yh)
	delta=yh-y
	out=yh
	return out,loss,delta

def train():
	W1 = np.random.uniform(low=0.0001, high=0.001, size=(27,300))
	W2 = np.random.uniform(low=0.0001, high=0.001, size=(300,50))
	W3 = np.random.uniform(low=0.0001, high=0.001, size=(50,1))
	W4 = np.random.uniform(low=0.0001, high=0.001, size=(200000,1))
	B1 = 0
	B2 = 0
	B3 = 0
	B4 = 0

	for k in range(0,1000,10):
		o1,b=batch(k,10)
		j=0
		for j in range(0,10):
			Lay_1,del_1=Rel(conv(W1,B1,b))
			Lay_2,del_2=Rel(conv2(W2,B2,Lay_1))
			Lay_3,del_3=Rel(conv2(W3,B3,Lay_2))
			Lay_4=np.einsum('ijk,ij->j',Lay_3,W4) + B4
			out,loss,delta=activ(Lay_4,o1)

			db4=delta
			dw4=np.reshape(np.dot(Lay_3[:,:,0],delta)/10,(200000,1))
			dl4=np.reshape(np.multiply(delta,W4),(200000,10,1))
			W4=W4-dw4
			B4=B4-db4

			d4= dl4*del_3
			db4=d4
			dw4=np.reshape(np.transpose(np.tensordot(d4,Lay_2,axes=([0,1],[0,1]))/10),(50,1))
			dl4=np.tensordot(d4,W3,axes=([2],[1]))
			W3=W3-dw4
			B3=B3-db4
			
			d4= dl4*del_2
			db4=d4
			dw4= np.reshape(np.transpose(np.tensordot(d4,Lay_1,axes=([0,1],[0,1]))/10),(300,50))
			dl4= np.tensordot(d4,W2,axes=([2],[1]))
			W2=W2-dw4
			B2=B2-db4

			d4= dl4*del_1
			db4= d4
			dw4= np.reshape(np.transpose(np.tensordot(d4,b,axes=([0,1],[0,2]))/10),(27,300))
			W1=W1-dw4
			B1=B1-db4

			print(np.sum(loss))
			j+=1
		print(k)
	print(out)
train()