import sys
import cv2
import numpy as np
import pandas as pd
import argparse
#from pyrsistent import T
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer
from torch import equal
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold


class DataSet():

	def __init__(self, tag, cross_val_num=10, portion=10):

		self.DFdata = pd.read_excel("./dataset/sample_data.xlsx", sheet_name = "sample") #"grasped" or "data_old"
		self.DFdata_0 = pd.read_excel("./dataset/sample_data.xlsx", sheet_name = "sample_0")
		self.DFdata_1 = pd.read_excel("./dataset/sample_data.xlsx", sheet_name = "sample_1")
		self.DFdata_2 = pd.read_excel("./dataset/sample_data.xlsx", sheet_name = "sample_2")
		#self._testData = pd.read_excel("./dataset/dataset.xlsx", sheet_name = "test")
		self._cross_val_num = cross_val_num
		self._portion = portion
		self._tag = tag
	
	def data(self, datasheet): 
		
		dataAttr   = datasheet[["LGPos", "RGPos", "Tpos"]]
		dataImgAdd = datasheet[["imgAdd"]]
		dataLabels = datasheet[[self._tag]] #"LSlip", "RSlip", "success", "output", "notGrasped"

		return dataAttr.to_numpy(), dataImgAdd.imgAdd.tolist(), dataLabels.to_numpy()

	def load_data(self, datasheet):

		dataAttr, dataAdds, dataLabels = self.data(datasheet)
		#trainAttr, trainAdds, trainLabels = self.train_data()
		#testAttr, testAdds, testLabels    = self.test()
		#print(len(dataAdds))
		imageHeight = 224
		imageWidth = 224
		dataImg  = []
		#testImg  = []
		
		#attributes = np.append(trainAttr, testAttr, axis=0)
		#maxVals = np.concatenate(([dataAttr.max(axis=0)],[testAttr.max(axis=0)]),axis=0).max(axis=0)
		#minVals = np.concatenate(([dataAttr.min(axis=0)],[testAttr.min(axis=0)]),axis=0).min(axis=0)

		maxVals = dataAttr.max(axis=0)
		minVals = dataAttr.min(axis=0)

		#print (type(maxVals))
		#print(maxVals.shape)
		dataAttr = (dataAttr - minVals)/(maxVals - minVals)
		#testAttr = (testAttr - minVals)/(maxVals - minVals)
		#trainAttr = (trainAttr - minVals)/(maxVals - minVals)

		#np.savetxt('./dataset/train.csv', trainAttr, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)
		#np.savetxt('./dataset/test.csv', testAttr, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ', encoding=None)
		#lb = LabelBinarizer()
		#dataLabels = lb.fit_transform(dataLabels).argmax(axis=1)
		#dataLabels = to_categorical(dataLabels)
		dataLabels = np.squeeze(dataLabels, axis=-1)
		#print(dataLabels[0])
		for i in range(0,len(dataAdds)):
			#print(dataAdds[i])
			d_img = cv2.imread(dataAdds[i])#, cv2.IMREAD_GRAYSCALE)
			#print(d_img.shape)
			d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
			d_img = cv2.resize(d_img, (imageWidth,imageHeight), interpolation=cv2.INTER_AREA)
			#tmp = d_img
			#cv2.putText(tmp, dataAdds[i][-9:-1], (5, 20),
					#cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
			#cv2.imshow("train", tmp)
			#cv2.waitKey(0)
			dataImg.append(d_img/255.0)
		
		"""
		for i in range(0,len(testAdds)):
			d_img = cv2.imread(testAdds[i][1:-1])#, cv2.IMREAD_GRAYSCALE)
			d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
			d_img = cv2.resize(d_img, (imageWidth,imageHeight))
			tmp = d_img
			cv2.putText(tmp, testAdds[i][-9:-1], (5, 20),
					cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
			cv2.imshow("train", tmp)
			cv2.waitKey(0)
			testImg.append(d_img/255.0)
		"""
		return ( np.array(dataAttr, dtype=np.float32), 
					np.array(dataImg, dtype=np.float32), 
					np.array(dataLabels, dtype=np.float32) )

	def split_data(self, file_name= './output/tmp_data.npy', data_load = -1):

		dataA, dataX, dataL = self.load_data(self.DFdata)

		tr_indx = []
		ts_indx = []
		num_tr_folds = []
		num_ts_folds = []
		img_tr_folds = []
		img_ts_folds = []
		lbl_tr_folds = []
		lbl_ts_folds = []
		cross_val_num = self._cross_val_num
		
		# Read from saved dataset (indices of K-folds) for test
		if data_load > 0:
			print('reading dataset...')
			with open(file_name, 'rb') as f:
				tr_indx = np.load(f, allow_pickle=True)
				_ = np.load(f, allow_pickle=True)
				_ = np.load(f, allow_pickle=True)
				_ = np.load(f, allow_pickle=True)
				ts_indx = np.load(f, allow_pickle=True)
				_ = np.load(f, allow_pickle=True)
				_ = np.load(f, allow_pickle=True)
				_ = np.load(f, allow_pickle=True)

			for K in range(0,cross_val_num):

				# select rows
				#print("test index: {}".format(test_ix))
				train_ix, test_ix = tr_indx[K], ts_indx[K]
				train_numerical, train_images, train_labels = dataA[train_ix], dataX[train_ix], dataL[train_ix]
				test_numerical, test_images, test_labels = dataA[test_ix], dataX[test_ix], dataL[test_ix]
				# Data Augmetnation
				# For data augmentation the below section is used to add different flips to the dataset
				"""
				for tmp_data in [self.DFdata_1]: #, self.DFdata_1, self.DFdata_2]:
					dataA_tmp, dataX_tmp, dataL_tmp = self.load_data(tmp_data)
					
					train_numerical = np.append(train_numerical, dataA_tmp[train_ix], axis=0)
					train_images = np.append(train_images, dataX_tmp[train_ix], axis=0)
					train_labels = np.append(train_labels, dataL_tmp[train_ix], axis=0)
					test_numerical = np.append(test_numerical, dataA_tmp[test_ix], axis=0)
					test_images = np.append(test_images, dataX_tmp[test_ix], axis=0)
					test_labels = np.append(test_labels, dataL_tmp[test_ix], axis=0)
				"""
				# summarize train and test composition
				train_0, train_1, train_2 = len(train_labels[train_labels==0]), len(train_labels[train_labels==1]), len(train_labels[train_labels==2])
				test_0, test_1, test_2 = len(test_labels[test_labels==0]), len(test_labels[test_labels==1]), len(test_labels[test_labels==2])
				print('>Train: 0=%d, 1=%d, 2=%d, Test: 0=%d, 1=%d, , 2=%d' % (train_0, train_1, train_2, test_0, test_1, test_2))


				num_tr_folds.append(train_numerical)
				num_ts_folds.append(test_numerical)
				img_tr_folds.append(train_images)
				img_ts_folds.append(test_images)
				lbl_tr_folds.append(train_labels)
				lbl_ts_folds.append(test_labels)


			return (tr_indx, num_tr_folds, img_tr_folds, lbl_tr_folds,
					ts_indx, num_ts_folds, img_ts_folds, lbl_ts_folds)
		
		if cross_val_num ==0:
			cross_val_num = 10
			portion = 10

		kfold = StratifiedKFold(n_splits=cross_val_num, shuffle=True)
		split_num = 1
		#print("portion = {}".format(portion))
		for train_ix, test_ix in kfold.split(dataA, dataL):
			# select rows
			#print("test index: {}".format(test_ix))
			#if (split_num == portion):
			train_numerical, train_images, train_labels = dataA[train_ix], dataX[train_ix], dataL[train_ix]
			test_numerical, test_images, test_labels = dataA[test_ix], dataX[test_ix], dataL[test_ix]
			# Data Augmetnation
			# For data augmentation the below section is used to add different flips to the dataset
			"""
			for tmp_data in [self.DFdata_1]: #, self.DFdata_1, self.DFdata_2]:
				dataA_tmp, dataX_tmp, dataL_tmp = self.load_data(tmp_data)
				
				train_numerical = np.append(train_numerical, dataA_tmp[train_ix], axis=0)
				train_images = np.append(train_images, dataX_tmp[train_ix], axis=0)
				train_labels = np.append(train_labels, dataL_tmp[train_ix], axis=0)
				test_numerical = np.append(test_numerical, dataA_tmp[test_ix], axis=0)
				test_images = np.append(test_images, dataX_tmp[test_ix], axis=0)
				test_labels = np.append(test_labels, dataL_tmp[test_ix], axis=0)
			"""
			# summarize train and test composition
			train_0, train_1, train_2 = len(train_labels[train_labels==0]), len(train_labels[train_labels==1]), len(train_labels[train_labels==2])
			test_0, test_1, test_2 = len(test_labels[test_labels==0]), len(test_labels[test_labels==1]), len(test_labels[test_labels==2])
			print('>Train: 0=%d, 1=%d, 2=%d, Test: 0=%d, 1=%d, , 2=%d' % (train_0, train_1, train_2, test_0, test_1, test_2))
			print(split_num)
			split_num += 1
			tr_indx.append(train_ix)
			ts_indx.append(test_ix)
			num_tr_folds.append(train_numerical)
			num_ts_folds.append(test_numerical)
			img_tr_folds.append(train_images)
			img_ts_folds.append(test_images)
			lbl_tr_folds.append(train_labels)
			lbl_ts_folds.append(test_labels)
		
		

		with open(file_name, 'wb') as f:
			np.save(f, np.array(tr_indx))
			np.save(f, np.array(num_tr_folds))
			np.save(f, np.array(img_tr_folds))
			np.save(f, np.array(lbl_tr_folds))
			np.save(f, np.array(ts_indx))
			np.save(f, np.array(num_ts_folds))
			np.save(f, np.array(img_ts_folds))
			np.save(f, np.array(lbl_ts_folds))

		return ( tr_indx, num_tr_folds, img_tr_folds, lbl_tr_folds,
					ts_indx, num_ts_folds, img_ts_folds, lbl_ts_folds)

	def train_data(self):
		(trA, trX, trL, _, _, _) = self.split_data()
		return (trA, trX, trL)

	def test_data(self):
		(_, _, _, tsA, tsX, tsL) = self.split_data()
		return (tsA, tsX, tsL)

"""
testNum = "00"
folds = 5
#RSlip
data_tag = 'LSlip' # data_tag = "LSlip" | "RSlip" | "success" | 'output'
file_name = "./output/cnn_{}_{}-{}".format(data_tag,testNum,folds)
DS = DataSet(tag=data_tag, cross_val_num=folds) # tag = "LSlip" | "RSlip" | "success"
(tr_indx, _, trX, trL, ts_indx, _, tsX, tsL) = DS.split_data(file_name , data_load = 1) # data_arg["load_model"]
tr_foldX, tr_foldY         = trX, trL
val_foldX, val_foldY       = tsX, tsL
ts_foldX, ts_foldY         = tsX, tsL

K = 1
print("[INFO] loading image data...")
#DS = DataSet(cross_val_num=folds, portion=K)
print( "train_ix:[{}...{}], test_ix:[{}...{}]".format(tr_indx[K-1][0:5], tr_indx[K-1][-6:-1], ts_indx[K-1][0:5], ts_indx[K-1][-6:-1]) )
trainX, trainY = tr_foldX[K-1][0:3], tr_foldY[K-1][0:3]
valX, valY     = val_foldX[K-1][0:3], val_foldY[K-1][0:3]
testX, testY   = ts_foldX[K-1], ts_foldY[K-1]
print (trainX.shape)
print (trainY.shape)
print (valX.shape)
print (valY.shape)
print (testX.shape)
print (testY.shape)
x = np.equal(testX[0],valX[0])
print(x[0:5][0:5])

y_tmp = trainY.flatten()
CLASS_WEIGHTS = class_weight.compute_class_weight('balanced', np.unique(trainY), y_tmp)

trainY = to_categorical(trainY)
valY = to_categorical(valY)
testY = to_categorical(testY)

#trainY = to_categorical(trL, num_classes=CLASS_NUM)
#valY = to_categorical(tsL, num_classes=CLASS_NUM)
print(np.array(ts_indx_0) == np.array(ts_indx_1))

ap1 = argparse.ArgumentParser()
ap1.add_argument("--plot", type=str, default="{}.png".format(save_name),
	help="path to output loss/accuracy plot") #"-p", 
ap1.add_argument("-m", "--model", type=str, default="{}.hdf5".format(file_name + "-1"),
	help="path to output loss/accuracy plot")
args = vars(ap1.parse_args())

print(trL.shape)
y_tmp = trL.flatten()
CLASS_WEIGHTS = class_weight.compute_class_weight('balanced',
												np.unique(trL),
												y_tmp)
print(CLASS_WEIGHTS)
print(trL[0])
trainY = to_categorical(trL)
print(type(trainY[0]))
cv2.imshow('test', tsX[0,:,:])
cv2.waitKey(0)

#print(trA.min(axis=0))
#print(trA.max(axis=0))
#print(tsA.min(axis=0))
#print(tsA.max(axis=0))
#print(trX.shape)
#print(trX[0])




indx =  [ np.where( trL == 0 )[0][0] , 
			np.where( trL == 1 )[0][0] ] 
#print(type(np.where( trL.argmax(axis=1) == 0 )))
print(type(indx))
print(indx)

print(trL[indx][0])
print(trA[indx][0])
print(trA[indx[0]])

print(trL[indx][1])
print(trA[indx][1])
#print(trX[indx[1]])

#print(trL[indx][2])
#print(trA[indx][2])
#print(trX[indx[2]])

		data_num = dataL.shape[0]
		#train_num = data_num - int(data_num / cross_val_num)
		val_num = int(data_num / cross_val_num)
		val_indx_range = list( range(0, data_num, val_num) )
		set_point = val_indx_range[portion]
		
		
		max_indx = np.max([len(np.where( dataL.argmax(axis=1) == 0 )[0]),
							len(np.where( dataL.argmax(axis=1) == 1 )[0]),
							len(np.where( dataL.argmax(axis=1) == 2 )[0])])
		indx = [np.where( dataL.argmax(axis=1) == 0 )[0][0], 
				np.where( dataL.argmax(axis=1) == 1 )[0][0],
				np.where( dataL.argmax(axis=1) == 2 )[0][0]]
		for i in range(1,max_indx):
			if i < len(np.where( dataL.argmax(axis=1) == 0 )[0]):
				indx.extend( [ np.where( dataL.argmax(axis=1) == 0 )[0][i]] )
			if i < len(np.where( dataL.argmax(axis=1) == 1 )[0]):
				indx.extend( [np.where( dataL.argmax(axis=1) == 1 )[0][i]] )
			if i < len(np.where( dataL.argmax(axis=1) == 2 )[0]):
				 indx.extend( [np.where( dataL.argmax(axis=1) == 2 )[0][i]] )
	
		print(indx)
		print(len(indx))
		print(type(indx))
		indx = np.array(indx)
		val_indx = list(range(0+set_point , val_num+set_point))
		indx_set = set(indx)
		train_indx = list(indx_set - set(indx[val_indx]))
		
		train_numerical = dataA[train_indx, :]
		train_images    = dataX[train_indx, :, :]
		train_labels    = dataL[train_indx, :]
		val_numerical   = dataA[indx[val_indx], :]
		val_images      = dataX[indx[val_indx], :, :]
		val_labels      = dataL[indx[val_indx], :]
		#test_numerical  = np.concatenate((dataA[test_indx[0]], dataA[test_indx[1]], dataA[test_indx[2]]), axis=0)
		#test_images     = np.concatenate((dataX[test_indx[0]], dataX[test_indx[1]], dataX[test_indx[2]]), axis=0)
		#test_labels     = np.concatenate((dataL[test_indx[0]], dataL[test_indx[1]], dataL[test_indx[2]]), axis=0)
		#test_numerical  = np.concatenate((dataA[test_indx[0]], dataA[test_indx[1]], dataA[test_indx[2]]), axis=0)
		#test_images     = np.concatenate((dataX[test_indx[0]], dataX[test_indx[1]], dataX[test_indx[2]]), axis=0)
		#test_labels     = np.concatenate((dataL[test_indx[0]], dataL[test_indx[1]], dataL[test_indx[2]]), axis=0)
		"""