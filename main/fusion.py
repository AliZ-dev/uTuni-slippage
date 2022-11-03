# import the necessary packages
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ast import arg
from unicodedata import name
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, TruePositives, TrueNegatives, FalsePositives, FalseNegatives
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import class_weight
from neuralnets import network
from readdata import DataSet
import matplotlib.pyplot as plt
import numpy as np
import argparse



testNum = "01"
folds = 5

data_tag = 'LSlip' # data_tag = "LSlip" | "RSlip" | "success" | 'output'
file_name = "./output/cnn_{}_{}-{}".format(data_tag,testNum,folds)
#file_name_data = "./output/cnn_{}_{}-{}".format(data_tag,testNum,folds)
DS = DataSet(tag=data_tag, cross_val_num=folds) # tag = "LSlip" | "RSlip" | "success"
(tr_indx, trA, trX, trL, ts_indx, tsA, tsX, tsL) = DS.split_data(file_name , data_load = 1) # data_arg["load_model"]
tr_foldA, tr_foldX, tr_foldY         = trA, trX, trL
val_foldA, val_foldX, val_foldY       = tsA, tsX, tsL
ts_foldA, ts_foldX, ts_foldY         = tsA, tsX, tsL

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
    # construct the argument parser and parse the arguments
    ap1 = argparse.ArgumentParser()
    ap1.add_argument("-l", "--load-model", type=int, default=-1,
        help="(optional) whether or not pre-trained model should be loaded")
    ap1.add_argument("--plot", type=str, default="{}.png".format(save_name),
        help="path to output loss/accuracy plot") #"-p", 
    ap1.add_argument("-m", "--model", type=str, default="{}.hdf5".format(save_name),
        help="path to output loss/accuracy plot")
    args = vars(ap1.parse_args())

    print("[INFO] loading image data...")
    #DS = DataSet(cross_val_num=folds, portion=K)
    print( "train_ix:[{}...{}], test_ix:[{}...{}]".format(tr_indx[K-1][0:5], tr_indx[K-1][-6:-1], ts_indx[K-1][0:5], ts_indx[K-1][-6:-1]) )
    trainA, trainX, trainY_bin = tr_foldA[K-1], tr_foldX[K-1], tr_foldY[K-1]
    valA, valX, valY     = val_foldA[K-1], val_foldX[K-1], val_foldY[K-1]
    testA, testX, testY_bin   = ts_foldA[K-1], ts_foldX[K-1], ts_foldY[K-1]
    print(tr_foldA[K-1].shape)
    y_tmp = trainY_bin.flatten()
    CLASS_WEIGHTS = class_weight.compute_class_weight('balanced', np.unique(trainY_bin), y_tmp)

    trainY = to_categorical(trainY_bin)
    valY = to_categorical(valY)
    testY = to_categorical(testY_bin)

    # initialize the initial learning rate, number of epochs to train for,
    # and batch size
    IMG_HEIGHT = trainX.shape[1]
    IMG_WIDTH  = trainX.shape[2]
    CLASS_NUM = trainY.shape[1]
    #print("number of classes = {}".format(CLASS_NUM))
    INIT_LR    = 1e-4
    EPOCHS     = 40
    BS         = 64
    print(trainA.shape)

    def cnn(CLASS_NUM, IMG_HEIGHT, IMG_WIDTH):

        base_model = network().cnn_layers(IMG_HEIGHT, IMG_WIDTH)
        #pos_in = Input(shape = (3,), name='position_input')
        #pos_out = Dense(8, activation="relu")(pos_in) # fusion_0, fusion_5
        print('VGG16_Ouptput_Shape:')
        print(base_model.output.shape)
        headModel = base_model.output
        #headModel = AveragePooling2D(pool_size=(2, 2))(headModel) 
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(100, activation="relu")(headModel)
        headModel = Dense(20, activation="relu")(headModel)
        #headModel = Concatenate()([headModel, pos_out]) # fusion_2, fusion_4, fusion_5
        headModel = Dropout(0.1)(headModel) # original=0.1, fusion_4 = fusion_5=0.2
        #headModel = Concatenate()([headModel, pos_out]) # fusion_3
        headModel = Dense(8, activation="relu")(headModel) # original=8
        #headModel = Concatenate()([headModel, pos_out]) # fusion_1
        headModel = Dense(CLASS_NUM, activation="softmax")(headModel)
        
        #model = Model(inputs=[base_model.input, pos_in], outputs=headModel)
        model = Model(inputs=base_model.input, outputs=headModel)

        print('VGG16_Layers_Output:')
        for layer in base_model.layers:
            layer.trainable = False
        
        return model

    model = cnn(CLASS_NUM, IMG_HEIGHT, IMG_WIDTH)
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR, beta_1=0.9, beta_2=0.99, epsilon=1e-08)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy", Precision(), Recall()]) # , Precision(), Recall()
        
    checkpoint = ModelCheckpoint("{}.hdf5".format(save_name), monitor='val_loss',
                    verbose=1, save_best_only=True, mode='auto') # , period=1
    earlystopper = EarlyStopping(patience=8, verbose=1)
    print(model.summary())

    if args["load_model"] < 0:
        # train the head of the network
        #model = load_model(args["model"])
        print("[INFO] training head...")
        H = model.fit(trainX, trainY, batch_size=BS,
            validation_data=(valX, valY), epochs=EPOCHS, validation_steps=1,
            callbacks=[checkpoint, earlystopper], verbose=1, class_weight=CLASS_WEIGHTS) # shuffle=False, validation_steps=1 class_weight=CLASS_WEIGHTS, 

        # plot the training loss and accuracy
        N = len(H.history["val_loss"])
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(args["plot"])

    else:
        model = load_model(args["model"])

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    img_probs = model.predict(testX)

    # fusion_06 >
    pos_model = SVC(probability=True, class_weight={0:CLASS_WEIGHTS[0], 1:CLASS_WEIGHTS[1]})
    #print(trainA.shape)
    pos_model.fit(trainA, trainY_bin)
    #print('after train')
    #print(pos_model)
    #expected = testY
    pos_probs = pos_model.predict_proba(testA)
    #print("pos_probs:{}".format(pos_probs.shape))
    probs = np.maximum(img_probs, pos_probs)
    # fusion_06 <


    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    prediction = np.argmax(probs, axis=1)
    # show a nicely formatted classification report
    print(classification_report(testY.argmax(axis=1), prediction, labels=list(range(0,CLASS_NUM)))) # target_names=np.array(['NotLSlip', 'LSlip'], dtype=object)'LSlip','RSlip', 'success'

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), prediction, labels=list(range(0,CLASS_NUM)))
    total = sum(sum(cm))
    #accuracy = (cm[0, 0] + cm[1, 1]) / total 
    #recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    #precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    #specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    #f1score = 2 * (precision * recall)/(precision + recall)
    accuracy = accuracy_score(testY.argmax(axis=1), prediction)
    print('\nAccuracy: {:.4f}\n'.format(accuracy_score(testY.argmax(axis=1), prediction)))
    print('Weighted Precision: {:.2f}'.format(precision_score(testY.argmax(axis=1), prediction, average='weighted')))
    print('Weighted Recall: {:.2f}'.format(recall_score(testY.argmax(axis=1), prediction, average='weighted')))
    print('Weighted F1-score: {:.2f}'.format(f1_score(testY.argmax(axis=1), prediction, average='weighted')))

    # show the confusion matrix, accuracy, sensitivity, and specificity
    cnf_mat[K-1,:,:] = cm
    print(cm)
    #print("accuracy: {:.4f}".format(accuracy))
    ##print("recall/sensitivity: {:.4f}".format(recall))
    #print("specificity: {:.4f}".format(specificity))
    acc.append(accuracy)
    #rec.append(recall)
    #prec.append(precision)
    #spec.append(specificity)
    #f1.append(f1score)

    count = 0
    for i in range(len(prediction)):

        #print("[INFO] probability: {}, Predicted: {}, Actual: {}".format(probs[i], prediction[i], testY[i]))
        if prediction[i] == testY[i].argmax():
            count += 1
        
    print("[INFO] Number of correct predictions: {}/{}".format(count,i+1))
    if args["load_model"] < 0:
        with open(save_name + "_history", 'wb') as f:
            np.save(f, np.array(H.history))
        
		

if args["load_model"] < 0:
	with open(file_name + "_acc", 'wb') as f:
		np.save(f, np.array(acc))
print("accuracy: \n {}".format(acc))
print("average accuracy: \n {}".format(np.mean(acc)))
#print("recall/sensitivity: \n {}".format(rec))
#print("precision: \n {}".format(prec))
#print("specificity: \n {}".format(spec))
#print("F1-score: \n {}".format(f1))
print("confusion matrix: \n {}".format(cnf_mat))
print("aggregated confusion matrix: \n {}".format(np.sum(cnf_mat, axis=0)))
"""
with open("{}.txt".format(file_name), "a") as f:
	f.write("\naccuracy: \n {}".format(acc))
	f.write("\nacc_avg: \n {}".format( np.mean(np.array(acc)) ) )
	#f.write("\nrecall/sensitivity: \n {}".format(rec))
	#f.write("\nrec_avg: \n {}".format( np.mean(np.array(rec)) ) )
	#f.write("\nprecision: \n {}".format(prec))
	#f.write("\nprec_avg: \n {}".format( np.mean(np.array(prec)) ) )
	#f.write("\nspecificity: \n {}".format(spec))
	#f.write("\nspec_avg: \n {}".format( np.mean(np.array(spec)) ) )
	#f.write("\nF1-score: \n {}".format(f1))
	#f.write("\nF1_avg: \n {}".format( np.mean(np.array(f1)) ) )
	f.write("\nconfusion matrix: \n {}".format(cnf_mat))
"""
"""
	f.write("\nFold: \n {}".format(K))
	f.write("\naccuracy: \n {}".format(H.history["val_accuracy"]))
	f.write("\nprecision: \n {}".format(H.history["val_precision"]))
	f.write("\nrecall: \n {}".format(H.history["val_recall"]))
	f.write("\ntruepositives: \n {}".format(H.history["val_true_positives"]))
	f.write("\ntruenegatives: \n {}".format(H.history["val_true_negatives"]))
	f.write("\nfalsepositives: \n {}".format(H.history["val_false_positives"]))
	f.write("\nfalsenegatives: \n {}".format(H.history["val_false_negatives"]))
"""