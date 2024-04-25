
import persim
from ripser import Rips
from sktime.datasets import load_from_tsfile
from sktime.datasets import load_from_arff_to_dataframe
from sktime.datasets import load_from_ucr_tsv_to_dataframe
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from  tqdm import tqdm
import  warnings
from persim.persistent_entropy import *
from sklearn.ensemble import RandomForestClassifier
warnings.simplefilter("ignore")
thresh=100
def VSPK(dgm):
    weights = dgm[:, 1] - dgm[:, 0]

    # 将权重与出生时间相乘
    weighted_births = weights * dgm[:, 0]

    # 计算加权均值
    weighted_mean = np.sum(weighted_births) / np.sum(weights)

    return weighted_mean
import xlrd
from xlutils.copy import copy
excel_path = 'F:\\Desktop\\CLAS\\class2.xlsx'  # 分类结果保存位置
excel_file = xlrd.open_workbook(excel_path)
workbook = copy(excel_file)
worksheet1 = workbook.get_sheet(sheet='1')





# train,test,length:
#1:20,20,512,2;  2:30,150,128,3;  3:60,60,577,4;  4:[50, 50, 270, 5]  5:[100, 100, 96, 2]
# 6:[109, 105, 431, 2],  7:60,60,448,3;  8:375,375,720,3  9: 181,77,900,2;
valuedit = {1:[1,512,5,2],2:[2,128,1,3],3:[3,577,5,4],4:[4, 270, 2, 5],5:[5, 96, 1, 2],
            6:[6, 431, 4, 2],7:[7,448,4,3],8:[8,720,7,3],9:[9,900, 9, 2]
            }#######数据长度，分类情况

pathdit = {1: 'BirdChicken',2: 'BME',3: 'Car',4: 'DuckDuckGeese', 5: 'ECG200', 6: 'Ham',
           7: 'Meat',8:'ScreenType', 9: 'WormsTwoClass'
             }#####数据名称
#23,24,25,54,55,68,38,20,9
"""
AAA1={}
AAA2={}
for xy in []:
    A1=F1Eall[xy]
    A2=a1Eall[xy]
    B=F1Rall[xy]
    A1 = np.array(A1)[np.argsort(B)[-1]]
    A2 = np.array(A2)[np.argsort(B)[-1]]
    AAA1[xy]=A1
    AAA2[xy]=A2

"""
F1Rall={}
a1Rall={}
F1Eall={}
a1Eall={}
import time

#9，30，35,5

for xy in [9]:#1,3,4,6,7,9,12,14,17,18,51,54,55,56,61,64,66,68,38,39,40,47,22,23,24,25,30,31,33,34,35,36,72,74,

    d = valuedit[xy][1] // 100
    if d == 0:
        d = 1
    # (wmin, wmax, wd) = (10*d, 11*d,d)
    (wmin, wmax, wd) = (35*d, 40*d , 5*d)
    path = pathdit[xy]
    value = valuedit[xy]

    filepath = 'E:\\m\\time_series\\2\\' + path + '\\' + path + '_'#####数据存储位置
    excel_line, w1, d1, class_number = value  # w窗口每个点云的点数-1，窗口大小#d窗口的间隔
    try:
        sample_tr, label_tr = load_from_tsfile(filepath + 'TRAIN.ts')
        sample_te, label_te = load_from_tsfile(filepath + 'TEST.ts')
    except:
        sample_tr, label_tr = load_from_arff_to_dataframe(filepath + 'TRAIN.arff')
        sample_te, label_te = load_from_arff_to_dataframe(filepath + 'TEST.arff')
    sample_tr = np.array(list(map(lambda x: x[0], np.array(sample_tr))))
    sample_te = np.array(list(map(lambda x: x[0], np.array(sample_te))))
    features_number = sample_tr.shape[1]
    sample_trnumber = sample_tr.shape[0]
    sample_tenumber = sample_te.shape[0]
    min_tr = min(list(map(lambda x: min(x), sample_tr)))
    max_tr = max(list(map(lambda x: max(x), sample_tr)))
    min_te = min(list(map(lambda x: min(x), sample_te)))
    max_te = max(list(map(lambda x: max(x), sample_te)))
    max = max(max_te, max_tr)
    min = min(min_te, min_te)
    del max_te, max_tr, min_te, min_tr
    sample_tr = (sample_tr - min) / (max - min)
    sample_te = (sample_te - min) / (max - min)
    del min, max
    rips = Rips(maxdim=1, thresh=thresh)

    F1all1=[]
    a1all1=[]
    F1all2 = []
    a1all2 = []
    worksheet1.write(8 * xy, 0, xy)
    worksheet1.write(8 * xy, 1, path)
    worksheet1.write(8 * xy, 2, 'train-a1')
    worksheet1.write(8 * xy + 1, 0, xy)
    worksheet1.write(8 * xy + 1, 1, path)
    worksheet1.write(8 * xy + 1, 2, 'train-f1')
    worksheet1.write(8 * xy + 2, 0, xy)
    worksheet1.write(8 * xy + 2, 1, path)
    worksheet1.write(8 * xy + 2, 2, 'test-a1')
    worksheet1.write(8 * xy + 3, 0, xy)
    worksheet1.write(8 * xy + 3, 1, path)
    worksheet1.write(8 * xy + 3, 2, 'test-f1')
    worksheet1.write(8 * xy + 4, 0, xy)
    worksheet1.write(8 * xy + 4, 1, path)
    worksheet1.write(8 * xy + 4, 2, 'time')
    worksheet1.write(8 * xy + 5, 0, xy)
    worksheet1.write(8 * xy + 5, 1, path)
    worksheet1.write(8 * xy + 5, 2, 'yuan-a1')
    worksheet1.write(8 * xy + 6, 0, xy)
    worksheet1.write(8 * xy + 6, 1, path)
    worksheet1.write(8 * xy + 6, 2, 'TS1-a1')
    worksheet1.write(8 * xy + 7, 0, xy)
    worksheet1.write(8 * xy + 7, 1, path)
    worksheet1.write(8 * xy + 7, 2, 'TS2-a1')

    for w in tqdm(range(wmin, wmax, wd),position=0):
        # w = int(ww*w1/100)
        t1=time.time()
        pc_number = (features_number - w) // d  # 一个图像的点云个数
        A = np.array(list(range(pc_number)))
        Ztr = np.random.random((0,6*pc_number))
        Zte = np.random.random((0,6*pc_number))
        Ztr1 = np.random.random((0, 2 * pc_number))
        Zte1 = np.random.random((0, 2 * pc_number))
        Ztr2 = np.random.random((0, 2 * pc_number))
        Zte2 = np.random.random((0, 2 * pc_number))
        Ztr3 = np.random.random((0, 2 * pc_number))
        Zte3 = np.random.random((0, 2 * pc_number))
        for x in tqdm(range(sample_trnumber),position=0):
            Br1 = sample_tr[x][:-1]
            Br2 = (sample_tr[x][1:] - sample_tr[x][:-1]+1)/2
            Br = np.append([Br1], [Br2], axis=0)
            Z=np.array([])
            Z1=np.array([])
            Z2 = np.array([])
            Z3 = np.array([])
            for y in A:
                dgms = rips.fit_transform(((Br[:, d * y:d * y + w]*[np.array(range(w))+1]/w)).T)
                H0_dgm = dgms[0]
                H1_dgm = dgms[1]
                H_dgm = np.concatenate((H0_dgm, H1_dgm), axis=0)
                H_dgmr1 = np.array(list(filter(lambda x: x[1] != np.inf, H_dgm)))
                if len(H_dgmr1) == 0:
                    H_dgmr1 = np.array([[0, 0]])
                dgms = rips.fit_transform((Br[:, d * y:d * y + w]*[w-np.array(range(w))]/w).T)
                H0_dgm = dgms[0]
                H1_dgm = dgms[1]
                H_dgm = np.concatenate((H0_dgm, H1_dgm), axis=0)
                H_dgmr2 = np.array(list(filter(lambda x: x[1] != np.inf, H_dgm)))
                if len(H_dgmr2) == 0:
                    H_dgmr2 = np.array([[0, 0]])
                dgms = rips.fit_transform(Br[:, d * y:d * y + w].T)
                H0_dgm = dgms[0]
                H1_dgm = dgms[1]
                H_dgm = np.concatenate((H0_dgm, H1_dgm), axis=0)
                H_dgmr3 = np.array(list(filter(lambda x: x[1] != np.inf, H_dgm)))
                if len(H_dgmr3) == 0:
                    H_dgmr3 = np.array([[0, 0]])
                AA=np.append(VSPK(H_dgmr1),disM.VSPK(H_dgmr2))
                AA=np.append(AA,VSPK(H_dgmr3))
                Z = np.append(Z,AA)
                Z1 = np.append(Z1,VSPK(H_dgmr3))
                Z2 = np.append(Z2, VSPK(H_dgmr1))
                Z3 = np.append(Z3,VSPK(H_dgmr2))
                del dgms, H0_dgm, H1_dgm, H_dgmr1, H_dgmr2, H_dgmr3,AA
            Ztr = np.append(Ztr,[Z],axis=0)
            Ztr1 = np.append(Ztr1, [Z1], axis=0)
            Ztr2 = np.append(Ztr2, [Z2], axis=0)
            Ztr3 = np.append(Ztr3, [Z3], axis=0)
            del Z, Br, Br1, Br2,Z1,Z2,Z3
        for x in tqdm(range(sample_tenumber),position=0):
            Be1 = sample_te[x][:-1]
            Be2 = (sample_te[x][1:] - sample_te[x][:-1]+1)/2
            Be = np.append([Be1], [Be2], axis=0)
            Z=np.array([])
            Z1 = np.array([])
            Z2 = np.array([])
            Z3 = np.array([])
            for y in A:
                dgms = rips.fit_transform((Be[:, d * y:d * y + w]*[np.array(range(w))+1]/w).T)
                H0_dgm = dgms[0]
                H1_dgm = dgms[1]
                H_dgm = np.concatenate((H0_dgm, H1_dgm), axis=0)
                H_dgme1 = np.array(list(filter(lambda x: x[1] != np.inf, H_dgm)))
                if len(H_dgme1) == 0:
                    H_dgme1 = np.array([[0, 0]])
                dgms = rips.fit_transform((Be[:, d * y:d * y + w]*[w-np.array(range(w))]/w).T)
                H0_dgm = dgms[0]
                H1_dgm = dgms[1]
                H_dgm = np.concatenate((H0_dgm, H1_dgm), axis=0)
                H_dgme2 = np.array(list(filter(lambda x: x[1] != np.inf, H_dgm)))
                if len(H_dgme2) == 0:
                    H_dgme2 = np.array([[0, 0]])
                dgms = rips.fit_transform(Be[:, d * y:d * y + w].T)
                H0_dgm = dgms[0]
                H1_dgm = dgms[1]
                H_dgm = np.concatenate((H0_dgm, H1_dgm), axis=0)
                H_dgme3 = np.array(list(filter(lambda x: x[1] != np.inf, H_dgm)))
                if len(H_dgme3) == 0:
                    H_dgme3 = np.array([[0, 0]])
                AA = np.append(VSPK(H_dgme1), disM.VSPK(H_dgme2))
                AA = np.append(AA, VSPK(H_dgme3))
                Z = np.append(Z,AA)
                Z1 = np.append(Z1, VSPK(H_dgme3))
                Z2 = np.append(Z2, VSPK(H_dgme1))
                Z3 = np.append(Z3, VSPK(H_dgme2))
                del dgms, H0_dgm, H1_dgm, H_dgme1, H_dgme2, H_dgme3, AA
            Zte = np.append(Zte,[Z],axis=0)
            Zte1 = np.append(Zte1, [Z1], axis=0)
            Zte2 = np.append(Zte2, [Z2], axis=0)
            Zte3 = np.append(Zte3, [Z3], axis=0)
            del Z,Be,Be1,Be2,Z1,Z2,Z3
        classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
####################################
        label_test01=[]
        for x in tqdm(range(sample_trnumber),position=0):
            AN = np.array(range(sample_trnumber))
            classifier.fit(Ztr[AN!=x], label_tr[AN!=x])
            labelpr = classifier.predict([Ztr[x]])
            label_test01.append(labelpr)
            labels = np.array(list(set(label_te) | set(label_tr)))
        f1 = f1_score(label_tr, label_test01, labels=labels, average="macro")
        a1 = accuracy_score(label_tr, label_test01)
        del label_test01,labels,labelpr,AN
        t2=time.time()
        print(t2-t1)

        print('')
        print(xy, '%s,%d,%d' % (path, d, w), f1, a1)
########################################
        classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
        classifier.fit(Ztr, label_tr)
        label_test01 = classifier.predict(Zte)
        labels = np.array(list(set(label_te) | set(label_tr)))

        f2 = f1_score(label_te, label_test01, labels=labels, average="macro")
        a2 = accuracy_score(label_te, label_test01)
        print(xy, '%s,%d,%d' % (path, d, w), f2, a2)
        del label_test01, labels,classifier
###############################################
        classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
        classifier.fit(Ztr1, label_tr)
        label_test01 = classifier.predict(Zte1)
        labels = np.array(list(set(label_te) | set(label_tr)))
        a3 = accuracy_score(label_te, label_test01)
####################################
        classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
        classifier.fit(Ztr2, label_tr)
        label_test01 = classifier.predict(Zte2)
        labels = np.array(list(set(label_te) | set(label_tr)))
        a4 = accuracy_score(label_te, label_test01)
#######################################################
        classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
        classifier.fit(Ztr3, label_tr)
        label_test01 = classifier.predict(Zte3)
        labels = np.array(list(set(label_te) | set(label_tr)))
        a5 = accuracy_score(label_te, label_test01)
        kk = int(w // (5*d)) + 6
        del Zte,Ztr,Zte1,Zte2,Zte3,Ztr1,Ztr2,Ztr3

        worksheet1.write(8 * xy ,kk, a1)
        worksheet1.write(8 * xy + 1, kk, f1)
        worksheet1.write(8 * xy + 2, kk, a2)
        worksheet1.write(8 * xy + 3, kk, f2)
        worksheet1.write(8 * xy + 4, kk, t2-t1)
        worksheet1.write(8 * xy + 5, kk, a3)
        worksheet1.write(8 * xy + 6, kk, a4)
        worksheet1.write(8 * xy + 7, kk, a5)

        workbook.save(excel_path)









