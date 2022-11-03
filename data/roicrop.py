import os
import cv2
import numpy as np
import pandas as pd
import csv
import scipy.io as sio
from glob import glob as gg
import matplotlib.pyplot as plt





class ROI():
    def __init__(self, root = '.'):
        self.cwd = os.getcwd()
        if root is not '.': self.cwd = root
        self.imgset = gg(self.cwd + "\\raw\\0*.tiff")
        self.targetAdd = self.cwd + "\\raw"
        


    def init_crop(self,init_roi_indx="00", get_add = True):
        self.targetAdd = self.cwd + "\\ROI_" + init_roi_indx
        if get_add: return self.targetAdd
        ROI_dict = sio.loadmat(self.targetAdd + ".mat")
        roi = ROI_dict['ROI']
        #print(imgSet[0][-9:-1])
        #"""
        f = open(self.targetAdd + "_adds.csv", "w")
        writer = csv.writer(f)

        for add in self.imgset:
            img = cv2.imread(add)
            #crop_roi = cv2.rectangle(img, (roi[0,0],roi[0,1]), (roi[0,2],roi[0,3]), (255,0,0), 3) #(679,935), (1653,1099)
            cropped = img[roi[0,1]:roi[0,1]+roi[0,3], roi[0,0]:roi[0,0]+roi[0,2]]
            image_path = (self.targetAdd + "\\" + add[-9:]).rstrip('\n')
            image_path = (self.targetAdd + "\\" + add[-9:]).rstrip('\r')
            image_path = (self.targetAdd + "\\" + add[-9:]).rstrip('\r\n')
            image_path = (self.targetAdd + "\\" + add[-9:]).rstrip('\n\r')

            new_adds = np.expand_dims(np.array(image_path),axis=-1)
            
            cv2.imwrite(new_adds,cropped)
            writer.writerow(new_adds)
            #crop_roi = cv2.resize(img, (800,600))
            #cropped = cv2.resize(cropped, (800,600))
            #cv2.imshow("Image", crop_roi)
            #cv2.imshow("Cropped", cropped)
            #cv2.waitKey(5000)

        f.close()
        #"""
        return self.targetAdd
        
    def find_roi(self, imgAdd = "0042.tiff"):

        #imgPath = self.init_crop(get_add=True)
        #print(imgPath)
        sample = cv2.cvtColor(cv2.imread(imgAdd), cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(sample,thresh=int(255*0.75), maxval=255, type=cv2.THRESH_BINARY_INV)
        #print(bw.shape)
        colScan = np.sum(bw/255,axis=0)
        #print(np.min(colScan))
        roi_c = np.where(colScan < np.min(colScan)*1.5)
        #print(len(roi_c))
        #print(roi_c[0][-1])
        #print(len(roi_c[0]))
        cropX = bw[:,roi_c[0][0]:roi_c[0][-1]]
        #print(cropX.shape)
        tmp = np.zeros(colScan.shape)
        #print(tmp.shape)
        #print(colScan.shape)
        tmp[roi_c[0]] = 1
        """
        plt.subplot(4,1,1)
        plt.plot(colScan)
        plt.grid(visible=True, which='major', axis='both')
        plt.subplot(4,1,2)
        plt.plot(tmp)
        plt.grid(visible=True, which='major', axis='both')
        """
        rowScan = np.sum(cropX/255,axis=1)
        #print(np.min(colScan))
        roi_r = np.where(rowScan > np.max(rowScan)*0.7)
        
        tmp = np.zeros(rowScan.shape)
        #print(tmp.shape)
        #print(rowScan.shape)
        tmp[roi_r[0]] = 1
        #print(roi_r[0].shape)
        """
        plt.subplot(4,1,3)
        plt.plot(rowScan)
        plt.grid(visible=True, which='major', axis='both')
        plt.subplot(4,1,4)
        plt.plot(tmp)
        plt.grid(visible=True, which='major', axis='both')
        plt.show()
        """
        #print(len(roi_r[0]))
        fiber_center = (int((roi_c[0][0]+roi_c[0][-1])/2), int((roi_r[0][0]+roi_r[0][-1])/2))
        #margin = 0
        #crop = sample[roi_r[0][0]-margin:roi_r[0][-1]+margin,roi_c[0][0]-margin:roi_c[0][-1]+margin]
        #crop_roi = cv2.rectangle(sample, (roi_c[0][0]-margin,roi_r[0][0]-margin), (roi_c[0][-1]+margin,roi_r[0][-1]+margin), (0,0,0), 3) #(679,935), (1653,1099)
        #crop_roi = cv2.circle(sample, fiber_center, 1, (0,0,0), 1)

        #crop_roi = cv2.resize(crop_roi, (800,600))
        #cropX = cv2.resize(cropX, (800,600))
        #cv2.imshow("roi", crop_roi)
        #cv2.imshow("fiber", cropX)
        #if cv2.waitKey(0) == 27:
        #    print("Quit")
        
        return fiber_center, roi_c[0], roi_r[0]

    def crop_to(self, ROI_indx = "01", crop_ratio = 0.1):
        target_path = self.cwd + "\\ROI_" + ROI_indx
        os.mkdir(target_path)
        dataset_path = self.cwd + "\\ROI_00" # Original data is in th "ROI_00" directory
        
        imgset = gg(dataset_path + "\\0*.tiff")
        dataset = pd.read_excel(dataset_path + "\\data_ROI_00.xlsx", sheet_name = "slippage")

        manual_ratio = 4
        #add = "0000.tiff"
        outputDF = pd.DataFrame(columns=dataset.columns)
        #print(dataset[["imgAdd"]].imgAdd.tolist())
        
        for i, add in zip(range(len(dataset[["imgAdd"]])), dataset[["imgAdd"]].imgAdd.tolist()):
            img = cv2.imread(add)
            #print(i)
            fiber_center, roi_c, roi_r = self.find_roi(imgAdd=add) # dataset.loc[0,'imgAdd']
            ratioX = (fiber_center[0]-roi_c[0])/fiber_center[0]
            #fiberRatio = (roi_r[-1]-roi_r[0])/(roi_c[-1]-roi_c[0]) * manual_ratio

            half_marginX = int(fiber_center[0] * (crop_ratio * (1-ratioX) + ratioX))
            half_marginY = int(crop_ratio * 500) # int(half_marginX * fiberRatio), 49
            cropped = img[fiber_center[1]-half_marginY:fiber_center[1]+half_marginY ,
                            fiber_center[0]-half_marginX:fiber_center[0]+half_marginX]
            si = add.rfind('.')
            #print(add[si-5:si])
            tmp_add = target_path + add[si-5:si] + ".tiff"
            outputDF = outputDF.append(pd.DataFrame([dataset.values[i]], columns= dataset.columns), ignore_index= True)
            outputDF.loc[i,'imgAdd'] = tmp_add
            cv2.imwrite(tmp_add, cropped)

        with pd.ExcelWriter(target_path + "\\data_ROI_" + ROI_indx + ".xlsx", mode='w', if_sheet_exists= 'overlay') as writer:
            outputDF.to_excel(writer, sheet_name='sample')
            
        
        
        
        #cv2.imshow("fiber", cropped)
        #if cv2.waitKey(0) == 27:
        #    print("Quit")









class DataAugment():
    def __init__(self, subject_path='.\\ROI_01', output_path='sample'):
        self._data = pd.read_excel(subject_path + "\\data_" + subject_path[2:] + ".xlsx", sheet_name = "sample")
        self.outputAdd = output_path
        os.mkdir(output_path)


    def image_flip(self):

        data = self._data
        #print(data.columns[:])
        dataImgAdd = data.imgAdd.tolist()
        outputDF = pd.DataFrame(columns=data.columns)
        data_0 = pd.DataFrame(columns=data.columns)
        data_1 = pd.DataFrame(columns=data.columns)
        data_2 = pd.DataFrame(columns=data.columns)
        for i, add in zip(range(len(dataImgAdd)), dataImgAdd):
            #print(i)
            #print(add)

            img = cv2.cvtColor(cv2.imread(add), cv2.COLOR_BGR2GRAY)
            flipVertical = cv2.flip(img, 0)
            flipHorizontal = cv2.flip(img, 1)
            flipBoth = cv2.flip(img, -1)

            si = add.rfind('ROI')
            tmp_add = add[:si] + self.outputAdd
            #print(tmp_add)
            si = add.rfind('.')
            add_0 = tmp_add + "\\" + add[si-4:si] + "_0.tiff"
            add_1 = tmp_add + "\\" + add[si-4:si] + "_1.tiff"
            add_2 = tmp_add + "\\" + add[si-4:si] + "_2.tiff"

            tmp_data = pd.DataFrame(columns=data.columns)
            #print(tmp_data.size)
            #print(len(list(data.values[i])))
            tmp_data = tmp_data.append(pd.DataFrame([data.values[i]], columns=data.columns), ignore_index=True)
            tmp_data.loc[0,'imgAdd'] = tmp_add + "\\" + add[si-4:si] + ".tiff"
            #tmp_data = pd.concat(pd.DataFrame(list(data.values[i]), columns=data.columns))
            #print(tmp_data.values)
            outputDF = outputDF.append(tmp_data, ignore_index=True)
            #print(outputDF.values)
            
            data_0 = data_0.append(tmp_data, ignore_index=True)
            data_0.loc[i,'imgAdd'] = add_0
            ##outputDF = outputDF.append(pd.DataFrame([data_0.values[i]], columns=data.columns), ignore_index=True) 

            
            data_1 = data_1.append(tmp_data, ignore_index=True)
            data_1.loc[i,'imgAdd'] = add_1
            data_1.loc[i,'LGPos'] = data.loc[i,'RGPos']
            data_1.loc[i,'RGPos'] = data.loc[i,'LGPos']
            data_1.loc[i,'LSlip'] = data.loc[i,'RSlip']
            data_1.loc[i,'RSlip'] = data.loc[i,'LSlip']
            if data.loc[i,'output'] == 1:
                #print('output=1')
                data_1.loc[i,'output'] = 2
                data_1.loc[i,'label'] = 'RSlip'
            elif data.loc[i,'output'] == 2:
                data_1.loc[i,'output'] = 1
                data_1.loc[i,'label'] = 'LSlip'

            ##outputDF = outputDF.append(pd.DataFrame([data_1.values[i]], columns=data.columns), ignore_index=True) 

            
            data_2 = data_2.append(tmp_data, ignore_index=True)
            data_2.loc[i,'imgAdd'] = add_2
            data_2.loc[i,'LGPos'] = data.loc[i,'RGPos']
            data_2.loc[i,'RGPos'] = data.loc[i,'LGPos']
            data_2.loc[i,'LSlip'] = data.loc[i,'RSlip']
            data_2.loc[i,'RSlip'] = data.loc[i,'LSlip']
            if data.loc[i,'output'] == 1:
                data_2.loc[i,'output'] = 2
                data_2.loc[i,'label'] = 'RSlip'
            elif data.loc[i,'output'] == 2:
                data_2.loc[i,'output'] = 1
                data_2.loc[i,'label'] = 'LSlip'

            ##outputDF = outputDF.append(pd.DataFrame([data_2.values[i]], columns=data.columns), ignore_index=True) 

            #print(tmp_data.loc[i,'imgAdd'])
            cv2.imwrite(tmp_data.loc[0,'imgAdd'], img)
            cv2.imwrite(add_0, flipVertical)
            cv2.imwrite(add_1, flipHorizontal)
            cv2.imwrite(add_2, flipBoth)

 
        with pd.ExcelWriter(tmp_add + "\\sample_data.xlsx", mode='w', if_sheet_exists= 'overlay') as writer:
            outputDF.to_excel(writer, sheet_name='sample')
            data_0.to_excel(writer, sheet_name='sample_0')
            data_1.to_excel(writer, sheet_name='sample_1')
            data_2.to_excel(writer, sheet_name='sample_2')




def main():
    ROIobj = ROI()
    ROIobj.crop_to(ROI_indx = "05", crop_ratio=0.4) # selecting different ROIs from original dataset and centering by middle of the fiber
    #ROIobj.find_roi() # for debugging purposes
    AugObj = DataAugment(subject_path='.\\ROI_05', output_path='sample_roi_05') # generating augmented dataset by different flips of images
    AugObj.image_flip()



if __name__=='__main__':
    main()