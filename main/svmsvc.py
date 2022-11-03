# Logistic Regression
from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import numpy as np
from readdata import DataSet



testNum = "test"
folds = 5
#RSlip
data_tag = 'success' # data_tag = "LSlip" | "RSlip" | "success" | 'output'
file_name = "./output/svm_{}_{}-{}".format(data_tag,testNum,folds)
DS = DataSet(tag=data_tag, cross_val_num=folds) # tag = "LSlip" | "RSlip" | "success"
(tr_indx, trX, _, trL, ts_indx, tsX, _, tsL) = DS.split_data(file_name , data_load = 0) # data_arg["load_model"]
tr_foldX, tr_foldY         = trX, trL
val_foldX, val_foldY       = tsX, tsL
ts_foldX, ts_foldY         = tsX, tsL

cnf_mat = np.zeros((folds,2,2))
acc  = []
spec = []
rec  = []
prec = []
f1 = []
start_fold = 1

for K in range(start_fold,folds+1):
    print("Fold = {}".format(K))
    save_name = "{}-{}".format(file_name,K)


    print("[INFO] loading data...")
    print( "train_ix:[{}...{}], test_ix:[{}...{}]".format(tr_indx[K-1][0:5], tr_indx[K-1][-6:-1], ts_indx[K-1][0:5], ts_indx[K-1][-6:-1]) )
    trainX, trainY = tr_foldX[K-1], tr_foldY[K-1]
    valX, valY     = val_foldX[K-1], val_foldY[K-1]
    testX, testY   = ts_foldX[K-1], ts_foldY[K-1]
    y_tmp = trainY.flatten()
    CLASS_WEIGHTS = class_weight.compute_class_weight('balanced', np.unique(trainY), y_tmp)
    #print(CLASS_WEIGHTS.shape)
    print(trainX.shape)
    model = SVC(probability=True, class_weight={0:CLASS_WEIGHTS[0], 1:CLASS_WEIGHTS[1]})
    #print('before train')
    model.fit(trainX, trainY)
    #print('after train')
    print(model)
    expected = testY
    predicted = model.predict(testX)
    print(predicted)
    print(model.predict_proba(testX))
    print(model.decision_function(testX))
    
    print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))
    accuracy = metrics.accuracy_score(expected, predicted)
    acc.append(accuracy)
    cnf_mat[K-1,:,:] = metrics.confusion_matrix(expected, predicted)

print("confusion matrix: \n {}".format(cnf_mat))
print("aggregated confusion matrix: \n {}".format(np.sum(cnf_mat, axis=0)))
print("accuracy: \n {}".format(acc))
print("average accuracy: \n {}".format(np.mean(acc)))