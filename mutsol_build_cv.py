import numpy as np 
import os 
import re
import sys 
import glob 
import pandas as pd

option = sys.argv[1]
num = int(sys.argv[2])
folder = sys.argv[3]

if folder == "ori":
    if not os.path.exists('./mutsol/inputs/'):
        os.mkdir('./mutsol/inputs')
    os.chdir('./mutsol/feature/')
    ss = ''
else:
    
    if not os.path.exists('./mutsol/inputs_'+folder+'/'):
        os.mkdir('./mutsol/inputs_'+folder+'/')
    os.chdir('./mutsol/feature_'+folder+'/')
    ss = '_'+folder

train_dataset = np.load("../idx/fold{}_train_pos.npy".format(num), allow_pickle=True)
test_dataset = np.load("../idx/fold{}_test.npy".format(num), allow_pickle=True)

mutsol = []
mutsol_Lap = []
mutsol_tf = []
mutsol_all = []
tf_test = []
Y_si = []
for i in range(len(train_dataset)):
    tmp = train_dataset[i]
    PDBid, Chains, muteChain, resID, resWT, resMT, si, temp, pH = str(tmp[0]), 'A', 'A', tmp[1][1:-1], tmp[1][0], tmp[1][-1], str(tmp[2]), 25, 7

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

    #print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT

    try:
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

        #print(np.shape(mutsol))
        #print(np.shape(mutsol_Lap))
        #print(np.shape(mutsol_tf))
        #print(np.shape(mutsol_all))
        #print(np.shape(Y_si))
        #print(np.shape(tf_test))
        os.chdir('..')
    except:
        print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
        os.chdir('..')
        continue

np.save('../inputs'+ss+'/fold{}_Y_train_pos.npy'.format(num), Y_si)
np.save('../inputs'+ss+'/fold{}_X_train_pos.npy'.format(num), mutsol)
np.save('../inputs'+ss+'/fold{}_Lap_train_pos.npy'.format(num), mutsol_Lap)
np.save('../inputs'+ss+'/fold{}_tf_train_pos.npy'.format(num), mutsol_tf)
np.save('../inputs'+ss+'/fold{}_all_train_pos.npy'.format(num), mutsol_all)
np.save('../inputs'+ss+'/fold{}_Transformer_train_pos.npy'.format(num), tf_test)

print(np.shape(mutsol_tf), np.shape(Y_si))

#print(np.var(tf_test, axis=0))


mutsol = []
mutsol_Lap = []
mutsol_tf = []
mutsol_all = []
tf_test = []
Y_si = []
for i in range(len(test_dataset)):
    tmp = test_dataset[i]
    PDBid, Chains, muteChain, resID, resWT, resMT, si, temp, pH = str(tmp[0]), 'A', 'A', tmp[1][1:-1], tmp[1][0], tmp[1][-1], str(tmp[2]), 25, 7

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

    if tmp[-1] == "rev":
        os.chdir('./'+PDBid+'_'+Chains+'_'+resMT+resID+resWT+'/')
    elif tmp[-1] == "pos":
        os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')

    #print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT

    try:
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

        #print(np.shape(mutsol))
        #print(np.shape(mutsol_Lap))
        #print(np.shape(mutsol_tf))
        #print(np.shape(mutsol_all))
        #print(np.shape(Y_si))
        #print(np.shape(tf_test))
        os.chdir('..')
    except:
        print('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
        os.chdir('..')
        continue

np.save('../inputs'+ss+'/fold{}_Y_test.npy'.format(num), Y_si)
np.save('../inputs'+ss+'/fold{}_X_test.npy'.format(num), mutsol)
np.save('../inputs'+ss+'/fold{}_Lap_test.npy'.format(num), mutsol_Lap)
np.save('../inputs'+ss+'/fold{}_tf_test.npy'.format(num), mutsol_tf)
np.save('../inputs'+ss+'/fold{}_all_test.npy'.format(num), mutsol_all)
np.save('../inputs'+ss+'/fold{}_test.npy'.format(num), tf_test)

print(np.shape(mutsol_tf), np.shape(Y_si))
#print(np.var(tf_test, axis=0))
