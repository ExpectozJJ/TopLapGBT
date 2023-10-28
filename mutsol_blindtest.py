import numpy as np 
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import scipy as sp
from math import sqrt
import pickle
import sys 
import os 
import pandas as pd
import multiprocessing as mp

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

n_estimators, lr, depth, subsample, mss = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
option = sys.argv[6]
folder = sys.argv[7]

if folder == "ori":
    os.chdir("./mutsol/")
    if not os.path.exists("./model/"):
        os.mkdir("./model/")
    ss = ''
else:
    ss = '_'+folder
    os.chdir("./mutsol/")
    if not os.path.exists("./model{}/".format(ss)):
        os.mkdir("./model{}/".format(ss))

def all_func(k):

    X_train, X_test = np.load("./inputs/fold{}_all_train_pos.npy".format(k), allow_pickle=True), np.load("./inputs/fold{}_all_test.npy".format(k), allow_pickle=True)
    y_train, y_test = np.load("./inputs/fold{}_Y_train_pos.npy".format(k), allow_pickle=True), np.load("./inputs/fold{}_Y_test.npy".format(k), allow_pickle=True)
    
    le = LabelEncoder()
    le.fit(['-1', '0', '1'])
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    reg = GradientBoostingClassifier(random_state=0, n_estimators = n_estimators, learning_rate=lr, max_features='sqrt', max_depth=depth, subsample=subsample, min_samples_split=mss)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    #p1 = reg.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    cc = gcc(y_test, y_pred)
    print("TopLapGBT Fold {}".format(k), acc, cc)
    cm = multilabel_confusion_matrix(y_test, y_pred)
    for i in range(3):
        tmp = cm[i]
        ppv, npv, sen, spec = tmp[1,1]/(tmp[1,1]+tmp[0,1]), tmp[0,0]/(tmp[0,0]+tmp[1,0]), tmp[1,1]/(tmp[1,1]+tmp[1,0]), tmp[0,0]/(tmp[0,0]+tmp[0,1])
        print(list(le.classes_)[i], ppv, npv, sen, spec)
    print("\n")

    pickle.dump(reg, open('./model/TopLapGBT_fold{}_{}_{}_{}_{}_{}.pkl'.format(k, n_estimators, lr, depth, subsample, mss), 'wb'))

def TopGBT(k, option):

    X_train, X_test = np.load("./inputs{}/fold{}_all_train_{}.npy".format(ss, k, option), allow_pickle=True), np.load("./inputs{}/fold{}_all_test.npy".format(ss,k), allow_pickle=True)
    y_train, y_test = np.load("./inputs{}/fold{}_Y_train_{}.npy".format(ss, k, option), allow_pickle=True), np.load("./inputs{}/fold{}_Y_test.npy".format(ss,k), allow_pickle=True)
    
    print(np.shape(X_train))

    le = LabelEncoder()
    le.fit(['-1', '0', '1'])
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    
    reg = GradientBoostingClassifier(random_state=42, n_estimators = n_estimators, learning_rate=lr, max_features='sqrt', max_depth=depth, subsample=subsample, min_samples_split=mss)
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
    print('TopGBT Fold {}: '.format(k) + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
    print(res)
    print('TopGBT Fold {} (normalized): '.format(k) + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
    print(res_b)

    pickle.dump(reg, open('./model{}/TopGBT_fold{}_{}_{}_{}_{}_{}_{}.pkl'.format(ss, k, n_estimators, lr, depth, subsample, mss, option), 'wb'))

for i in range(1,11):
    TopGBT(i, "pos")
    all_func(i)

Y = []
results = []
#grid_search()
for k in range(1,11):
    reg = pickle.load(open('./model/TopGBT_fold{}_{}_{}_{}_{}_{}_{}.pkl'.format(k, n_estimators, lr, depth, subsample, mss, option), 'rb'))

    X_test = np.load("./inputs/fold{}_all_test.npy".format(k), allow_pickle=True)
    y_test = np.load("./inputs/fold{}_Y_test.npy".format(k), allow_pickle=True)

    le = LabelEncoder()
    le.fit(['-1', '0', '1'])
    y_test = le.transform(y_test)
    
    y_pred = reg.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    y_pred = np.array(y_pred, dtype=int)
    y_test = le.inverse_transform(y_test)
    y_test = np.array(y_test, dtype=int)

    if len(Y) == 0:
        Y = y_test
        results = y_pred
    else:
        Y = np.concatenate((Y, y_test))
        results = np.concatenate((results, y_pred))

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
print('TopGBT 10-Fold CV: ' + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
print(res)
print('TopGBT 10-Fold CV (normalized): ' + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
print(res_b)

probs = []
labels = []
for k in range(1, 11):
    reg = pickle.load(open('./model/TopGBT_fold{}_{}_{}_{}_{}_{}_{}.pkl'.format(k, n_estimators, lr, depth, subsample, mss, option), 'rb'))

    X_test, y_test = np.load("./inputs/blind_all_test.npy", allow_pickle=True), np.load("./inputs/blind_Y_test.npy", allow_pickle=True)
    le = LabelEncoder()
    le.fit(['-1', '0', '1'])
    y_test = le.transform(y_test)

    y_pred = reg.predict(X_test)
    y_pred = le.inverse_transform(y_pred)
    y_pred = np.array(y_pred, dtype=int)
    y_test = le.inverse_transform(y_test)
    y_test = np.array(y_test, dtype=int)

    acc, gc2, res = gen_metrics(y_test, y_pred, balance=False, k=3)
    ## balance
    acc_b, gc2_b, res_b = gen_metrics(y_test, y_pred, balance=True, k=3)

    #f1 = open('results/fold'+str(fold+1)+'_val.txt', 'a')
    #print('TopGBT Blind Test Fold {}: '.format(k) + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
    #print(res)
    #print('TopGBT Blind Test (normalized) Fold {}: '.format(k) + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
    #print(res_b)

    p1 = reg.predict_proba(X_test)
    probs.append(p1)
    labels.append(y_pred)

probs = np.array(probs)
labels = np.array(labels)

### Soft Voting 
result_prob = np.mean(probs, axis=0)
y_pred = np.argmax(result_prob, axis=1)
y_pred = le.inverse_transform(y_pred)
y_pred = np.array(y_pred, dtype=int)

acc, gc2, res = gen_metrics(y_test, y_pred, balance=False, k=3)
## balance
acc_b, gc2_b, res_b = gen_metrics(y_test, y_pred, balance=True, k=3)

#f1 = open('results/fold'+str(fold+1)+'_val.txt', 'a')
print('TopGBT Soft Voting: ' + '| CPR:%.3f\t' % acc + '| GC2:{}\n'.format(gc2))
print(res)
print('TopGBT Soft Voting (normalized): ' + '| CPR:%.3f\t' % acc_b + '| GC2:{}\n'.format(gc2_b))
print(res_b)