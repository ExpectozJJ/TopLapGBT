import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import scipy as sp
from math import sqrt
import pickle
import sys 
import os 
import pandas as pd
import multiprocessing as mp
from sklearn.preprocessing import normalize

def gen_metrics(y_true, y_pre, balance=False, k=3):
    label = pd.DataFrame({'true': y_true, 'pre': y_pre})

    #print(label)

    unique_state = label.true.unique()
    targets = {}  
    ii = io = id_ = oi = oo = od = di = do = dd = 0
    tpi = tni = fpi = fni = tpd = tnd = fpd = fnd = tpo = tno = fpo = fno = 0
    for i, (t, p) in label.iterrows():
      
        if t == -1 and p == -1:
            dd += 1
        if t == -1 and p == 0:
            do += 1
        if t == -1 and p == 1:
            di += 1
        if t == 0 and p == -1:
            od += 1
        if t == 0 and p == 0:
            oo += 1
        if t == 0 and p == 1:
            oi += 1
        if t == 1 and p == -1:
            id_ += 1
        if t == 1 and p == 0:
            io += 1
        if t == 1 and p == 1:
            ii += 1

    alli = ii + io + id_
    alld = di + do + dd
    allo = oi + oo + od

  
    if balance:
        ii = ii * allo / alli
        io = io * allo / alli
        id_ = id_ * allo / alli
        di = di * allo / alld
        do = do * allo / alld
        dd = dd * allo / alld

    
    acc = (ii + oo + dd) / (ii + io + id_ + oi + oo + od + di + do + dd)
    N = ii + io + id_ + oi + oo + od + di + do + dd

 
    eii = (ii + io + id_) * (ii + oi + di) / N
    eio = (ii + io + id_) * (io + oo + do) / N
    eid_ = (ii + io + id_) * (id_ + od + dd) / N

    eoi = (oi + oo + od) * (ii + oi + di) / N
    eoo = (oi + oo + od) * (io + oo + do) / N
    eod = (oi + oo + od) * (id_ + od + dd) / N

    edi = (di + do + dd) * (ii + oi + di) / N
    edo = (di + do + dd) * (io + oo + do) / N
    edd = (di + do + dd) * (id_ + od + dd) / N

   
    gc2 = None
    if 0 not in [eii, eio, eid_, eoi, eoo, eod, edi, edo, edd]:
        gc2 = (((ii - eii) * (ii - eii) / eii) + ((io - eio) * (io - eio) / eio) +
               ((id_ - eid_) * (id_ - eid_) / eid_) + ((oi - eoi) * (oi - eoi) / eoi) +
               ((oo - eoo) * (oo - eoo) / eoo) + ((od - eod) * (od - eod) / eod) +
               ((di - edi) * (di - edi) / edi) + ((do - edo) * (do - edo) / edo) +
               ((dd - edd) * (dd - edd) / edd)) / ((k - 1) * N)

  
    seni = ii / (ii + io + id_)
    send = dd / (di + do + dd)
    seno = oo / (oi + oo + od)

  
    spei = (dd + do + od + oo) / (dd + do + od + oo + di + oi)
    sped = (ii + io + oi + oo) / (ii + io + oi + oo + id_ + od)
    speo = (dd + di + id_ + ii) / (dd + di + id_ + ii + do + io)

   
    ppvi = ppvd = ppvo = None
    if ii + oi + di != 0:
        ppvi = ii / (ii + oi + di)
    if id_ + dd + od != 0:
        ppvd = dd / (id_ + dd + od)
    if io + do + oo != 0:
        ppvo = oo / (io + do + oo)

    
    npvi = npvd = npvo = None
    if dd + do + od + oo + io + id_ != 0:
        npvi = (dd + do + od + oo) / (dd + do + od + oo + io + id_)
    if ii + io + oi + oo + di + do != 0:
        npvd = (ii + io + oi + oo) / (ii + io + oi + oo + di + do)
    if dd + di + id_ + ii + od + oi != 0:
        npvo = (dd + di + id_ + ii) / (dd + di + id_ + ii + od + oi)

    
    tpi = ii
    tni = oo + od + do + dd
    fpi = oi + di
    fni = io + id_

    tpd = dd
    fnd = di + do
    fpd = id_ + od
    tnd = ii + io + oi + oo

    tpo = oo
    fno = oi + od
    fpo = io + do
    tno = ii + id_ + di + dd
    columns = ['tp', 'tn', 'fp', 'fn', 'ppv', 'npv', 'tpr', 'tnr']
    res2 = pd.DataFrame(
        [
            [tpd, tnd, fpd, fnd, ppvd, npvd, send, sped],
            [tpo, tno, fpo, fno, ppvo, npvo, seno, speo],
            [tpi, tni, fpi, fni, ppvi, npvi, seni, spei]
        ],
        columns=columns,
        index=[-1, 0, 1]
    )
    return acc, gc2, res2

os.chdir("./mutsol/")

data = np.load("./mutsol.npy", allow_pickle=True)

def TopGBT(X):

    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    train_idx, test_idx = [], []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        _TopGBT(i, train_index, test_index)

def _TopGBT(k, train_index, test_index):

    X, y = np.load('./inputs/all_train_pos.npy', allow_pickle=True), np.load('./inputs/Y_train_pos.npy', allow_pickle=True)

    #X = normalize(X, axis=0)

    np.save('./model_random/fold{}_train_idx.npy'.format(k), train_index)
    np.save('./model_random/fold{}_test_idx.npy'.format(k), test_index)

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    print(np.shape(X_train))

    le = LabelEncoder()
    le.fit(['-1', '0', '1'])
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    reg = GradientBoostingClassifier(random_state=42, n_estimators = 20000, learning_rate=0.05, max_features='sqrt', max_depth=7, subsample=0.4, min_samples_split=3)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    y_pred = np.array(y_pred, dtype=int)
    y_test = le.inverse_transform(y_test)
    y_test = np.array(y_test, dtype=int)

    #p1 = reg.predict_proba(X_test)
    acc, gc2, res = gen_metrics(y_test, y_pred, balance=False, k=3)
    ## balance
    acc_b, gc2_b, res_b = gen_metrics(y_test, y_pred, balance=True, k=3)

    #f1 = open('results/fold'+str(fold+1)+'_val.txt', 'a')
    print('TopLapGBT Fold {}: '.format(k) + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
    print(res)
    print('TopLapGBT Fold {} (normalized): '.format(k) + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
    print(res_b)

    pickle.dump(reg, open('./model_random/TopLapGBT_fold{}_20000_0.05_7_0.4_3_pos.pkl'.format(k), 'wb'))

os.chdir('./feature/')

mutsol = []
mutsol_Lap = []
mutsol_tf = []
mutsol_all = []
tf_test = []
Y_si = []
af, topf = [], []

for i in range(len(data)):
    tmp = data[i]
    PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH = str(tmp[0]), tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], int(tmp[6]), tmp[7], tmp[8]

    #if PDBid != "149211803" and PDBid != "8248030":
    if PDBid[:4] == "1RTP":
        if Chains == 'A': 
            Chains = '1'
            muteChain = '1'
        elif Chains == 'B':
            Chains = '2'
            muteChain = '2'
        elif Chains == 'C':
            Chains = '3'
            muteChain = '3'

    #print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT
    os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')

    aux_feat = np.load(filename+'_aux.npy', allow_pickle=True)
    top_feat = np.load(filename+'_Top.npy', allow_pickle=True)
    lap_feat = np.load(filename+'_Lap.npy', allow_pickle=True)
    tmp = np.load(filename+'_Transformer.npy', allow_pickle=True)
    tmp2 = tmp[0]-tmp[1]
    tmp = list(tmp)
    tmp.append(tmp2)
    #tf_feat = np.reshape(tmp, (3*1280,))
    tf_feat = np.reshape(tmp, (3*1280,))
    tf_test.append(tf_feat)

    af.append(aux_feat)

    topf.append(top_feat)

    mutsol.append(np.concatenate((top_feat, aux_feat)))
    mutsol_Lap.append(lap_feat)
    mutsol_tf.append(np.concatenate((top_feat, aux_feat, tf_feat)))
    #print(np.shape(mutsol_tf))
    mutsol_all.append(np.concatenate((top_feat, aux_feat,lap_feat,tf_feat)))
    Y_si.append(sol) 
    os.chdir('../') 

np.save('../inputs/Y_train_pos.npy', Y_si)
np.save('../inputs/X_train_pos.npy', mutsol)
np.save('../inputs/Lap_train_pos.npy', mutsol_Lap)
np.save('../inputs/tf_train_pos.npy', mutsol_tf)
np.save('../inputs/all_train_pos.npy', mutsol_all)
np.save('../inputs/Transformer_train_pos.npy', tf_test)
np.save('../inputs/Top_train_pos.npy', topf)
np.save('../inputs/aux_train_pos.npy', af)

os.chdir('../')

X = np.load('./inputs/all_train_pos.npy', allow_pickle=True)

TopGBT(X)

X = np.load('./inputs/all_train_pos.npy', allow_pickle=True)
#X = normalize(X, axis=0)
Y = np.load('./inputs/Y_train_pos.npy', allow_pickle=True)
results = np.zeros(len(Y))

#grid_search()
for k in range(10):
    reg = pickle.load(open('./model_random/TopLapGBT_fold{}_20000_0.05_7_0.4_3_pos.pkl'.format(k), 'rb'))

    train_index = np.load('./model_random/fold{}_train_idx.npy'.format(k), allow_pickle=True)
    test_index = np.load('./model_random/fold{}_test_idx.npy'.format(k), allow_pickle=True)

    X_test = X[test_index]
    y_test = Y[test_index]

    #print(y_test)

    le = LabelEncoder()
    le.fit(['-1', '0', '1'])
    y_test = le.transform(y_test)
    
    y_pred = reg.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    y_pred = np.array(y_pred, dtype=int)
    y_test = le.inverse_transform(y_test)
    y_test = np.array(y_test, dtype=int)

    #print(y_pred)
    #print(y_test)

    results[test_index] = y_pred

    acc, gc2, res = gen_metrics(y_test, y_pred, balance=False, k=3)
    ## balance
    acc_b, gc2_b, res_b = gen_metrics(y_test, y_pred, balance=True, k=3)

    #print(res)

    #f1 = open('results/fold'+str(fold+1)+'_val.txt', 'a')
    #print('TopGBT Fold {}: '.format(k) + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
    #print(res)
    #print('TopGBT Fold {} (normalized): '.format(k) + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
    #print(res_b)

acc, gc2, res = gen_metrics(Y, results, balance=False, k=3)
## balance
acc_b, gc2_b, res_b = gen_metrics(Y, results, balance=True, k=3)

#f1 = open('results/fold'+str(fold+1)+'_val.txt', 'a')
print('TopLapGBT 10-Fold CV: ' + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
print(res)
print('TopLapGBT 10-Fold CV (normalized): ' + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
print(res_b)

