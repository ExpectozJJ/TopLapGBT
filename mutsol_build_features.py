import numpy as np 
import os 
import re
import sys 
import glob 
import pandas as pd


os.chdir('./mutsol/feature/')
train_dataset = pd.read_csv('../train_dataset.csv').to_numpy()
test_dataset = pd.read_csv('../test2_dataset.csv').to_numpy()

mutsol = []
mutsol_Lap = []
mutsol_tf = []
mutsol_all = []
tf_test = []
Y_si = []
for i in range(len(train_dataset)):
    tmp = train_dataset[i]
    PDBid, Chains, muteChain, resID, resWT, resMT, si, temp, pH = str(tmp[-1]), 'A', 'A', tmp[1][1:-1], tmp[1][0], tmp[1][-1], str(tmp[2]), 25, 7
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

    #print(PDBid, Chains, muteChain, resID, resWT, resMT, si, temp, pH)
    os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT

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
    mutsol.append(np.concatenate((top_feat, aux_feat)))
    mutsol_Lap.append(lap_feat)
    mutsol_tf.append(np.concatenate((top_feat, aux_feat, tf_feat)))
    #print(np.shape(mutsol_tf))
    mutsol_all.append(np.concatenate((top_feat, aux_feat,lap_feat,tf_feat)))
    Y_si.append(si)

    print(np.shape(mutsol))
    print(np.shape(mutsol_Lap))
    print(np.shape(mutsol_tf))
    print(np.shape(mutsol_all))
    print(np.shape(Y_si))
    print(np.shape(tf_test))
    os.chdir('..')

np.save('../mutsol_Y_train.npy', Y_si)
np.save('../mutsol_X_train.npy', mutsol)
np.save('../mutsol_Lap_train.npy', mutsol_Lap)
np.save('../mutsol_tf_train.npy', mutsol_tf)
np.save('../mutsol_all_train.npy', mutsol_all)
np.save('../Transformer_train.npy', tf_test)

#print(np.var(tf_test, axis=0))

mutsol = []
mutsol_Lap = []
mutsol_tf = []
mutsol_all = []
tf_test = []
Y_si = []
for i in range(len(test_dataset)):
    tmp = test_dataset[i]
    PDBid, Chains, muteChain, resID, resWT, resMT, si, temp, pH = str(tmp[-1]), 'A', 'A', tmp[1][1:-1], tmp[1][0], tmp[1][-1], str(tmp[2]), 25, 7
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

    #print(PDBid, Chains, muteChain, resID, resWT, resMT, si, temp, pH)
    os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT

    #try:
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
    mutsol.append(np.concatenate((top_feat, aux_feat)))
    mutsol_Lap.append(lap_feat)
    mutsol_tf.append(np.concatenate((top_feat, aux_feat, tf_feat)))
    #print(np.shape(mutsol_tf))
    mutsol_all.append(np.concatenate((top_feat, aux_feat,lap_feat,tf_feat)))
    Y_si.append(si)

    print(np.shape(mutsol))
    print(np.shape(mutsol_Lap))
    print(np.shape(mutsol_tf))
    print(np.shape(mutsol_all))
    print(np.shape(Y_si))
    print(np.shape(tf_test))
    os.chdir('..')
    #except:
        #print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
        #os.chdir('..')
        #continue

np.save('../inputs/blind_Y_test.npy', Y_si)
np.save('../inputs/blind_X_test.npy', mutsol)
np.save('../inputs/blind_Lap_test.npy', mutsol_Lap)
np.save('../inputs/blind_tf_test.npy', mutsol_tf)
np.save('../inputs/blind_all_test.npy', mutsol_all)
np.save('../inputs/blind_test.npy', tf_test)

#print(np.var(tf_test, axis=0))
