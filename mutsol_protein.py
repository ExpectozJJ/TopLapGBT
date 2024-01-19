import os, pickle, operator, sys, time
import numpy as np
import scipy as sp
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.Blast.Applications import NcbipsiblastCommandline
from scipy.spatial import cKDTree
import gudhi

FRIDefault = [['Lorentz', 0.5,  5],
              ['Lorentz', 1.0,  5],
              ['Lorentz', 2.0,  5],
              ['Exp',     1.0, 15],
              ['Exp',     2.0, 15]]
ele2index = {'C':0, 'N':1, 'O':2, 'S':3, 'H':4}
ss2index = {'H':1, 'E':2, 'G':3, 'S':4, 'B':5, 'T':6, 'I':7, '-':0}
Hydro = ['A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']
PolarAll = ['S','T','N','Q','R','H','K','D','E']
PolarUncharged = ['S','T','N','Q']
PolarPosCharged = ['R','H','K']
PolarNegCharged = ['D','E']
SpecialCase = ['C','U','G','P']
AAvolume = {'A': 88.6, 'R':173.4, 'D':111.1, 'N':114.1, 'C':108.5, \
            'E':138.4, 'Q':143.8, 'G': 60.1, 'H':153.2, 'I':166.7, \
            'L':166.7, 'K':168.6, 'M':162.9, 'F':189.9, 'P':112.7, \
            'S': 89.0, 'T':116.1, 'W':227.8, 'Y':193.6, 'V':140.0}
AAhydropathy = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5, \
                'E':-3.5, 'Q':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5, \
                'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6, \
                'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2}
AAarea = {'A':115., 'R':225., 'D':150., 'N':160., 'C':135., \
          'E':190., 'Q':180., 'G': 75., 'H':195., 'I':175., \
          'L':170., 'K':200., 'M':185., 'F':210., 'P':145., \
          'S':115., 'T':140., 'W':255., 'Y':230., 'V':155.}
AAweight = {'A': 89.094, 'R':174.203, 'N':132.119, 'D':133.104, 'C':121.154, \
            'E':147.131, 'Q':146.146, 'G': 75.067, 'H':155.156, 'I':131.175, \
            'L':131.175, 'K':146.189, 'M':149.208, 'F':165.192, 'P':115.132, \
            'S':105.093, 'T':119.12 , 'W':204.228, 'Y':181.191, 'V':117.148}
AApharma = {'A':[0,1,3,1,1,1],'R':[0,3,3,2,1,1],'N':[0,2,4,1,1,0],'D':[0,1,5,1,2,0],\
            'C':[0,2,3,1,1,0],'E':[0,1,5,1,2,0],'Q':[0,2,4,1,1,0],'G':[0,1,3,1,1,0],\
            'H':[0,3,5,3,1,0],'I':[0,1,3,1,1,2],'L':[0,1,3,1,1,1],'K':[0,2,4,2,1,2],\
            'M':[0,1,3,1,1,2],'F':[1,1,3,1,1,1],'P':[0,1,3,1,1,1],'S':[0,2,4,1,1,0],\
            'T':[0,2,4,1,1,1],'W':[2,2,3,1,1,2],'Y':[1,2,4,1,1,1],'V':[0,1,3,1,1,1]}
Groups = [Hydro, PolarAll, PolarUncharged, PolarPosCharged, PolarNegCharged, SpecialCase]

def atmtyp_to_ele( st ):
    if len(st.strip()) == 1:
        return st.strip()
    elif st[0] == 'H':
        return 'H'
    elif st == "CA":
        return "CA"
    elif st == "CL":
        return "CL"
    elif st == "BR":
        return "BR"
    else:
        print(st, 'Not in dictionary')
        return

def AAcharge(AA):
    if AA in ['D','E']:
        return -1.
    elif AA in ['R','H','K']:
        return 1.
    else:
        return 0.

class atom:
    def __init__(self, AType, AVType, Charge, Chain, ResName, ResID, Radii):
        self.pos         = None
        self.atype       = AType.replace(' ', '')
        self.verboseType = AVType
        self.Charge      = Charge
        self.ResName     = ResName
        self.ResID       = ResID
        self.R           = Radii
        self.Chain       = Chain
        self.Area        = 0.
        self.SolvEng     = 0.
    def position(self, pos):
        self.pos = pos

class Protein:
    def __init__(self, structure, typeFlag, onlyBLAST=False):
        self.PDBid    = structure.PDBid
        self.Chain    = structure.muteChain
        self.ResIDSeq = structure.resID_fasta # PSSM will use this
        self.typeFlag = typeFlag
        if typeFlag == 'WT':
            self.ResName  = structure.resWT
            self.Sequence = structure.fastaWT
        elif typeFlag == 'MT':
            self.ResName  = structure.resMT
            self.Sequence = structure.fastaMT
        else:
            sys.exit('wrong typeFlag for MutSol Proteins.py')

        self.filename = structure.PDBid+'_'+self.typeFlag
        self.filename_single = '_'.join([self.PDBid, self.Chain, self.typeFlag])

        if not onlyBLAST:
            self.ResID = structure.resID_MutedPartner
            if os.path.exists(self.filename+'.pqr'):
                self.loadPQRFile()
            if os.path.exists(self.filename+'.propka'):
                self.get_pka_info()
            self.IndexList = self.construct_index_list()
            self.setup_pairwise_interaction()

            self.SeqLength = len(structure.fasta['WT'])
        
    def loadPQRFile(self):
        print('load PQR file >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        ## Atom position from PQR file
        self.AtomPos = []
        self.Atoms   = []
        self.Charge  = []

        PQRFile = open(self.filename+'.pqr')
        for line in PQRFile:
            if line[0:4] == 'ATOM':
                resname = line[17:20]
                #if resname=='HSE': 
                #    resname='HIS'
                self.AtomPos.append([float(line[26:38]), float(line[38:46]), float(line[46:54])])
                Atom = atom(atmtyp_to_ele(line[12:14]), line[11:17], float(line[54:62]),
                            line[21], three_to_one(resname), int(line[22:26]), float(line[62:69]))
                self.Atoms.append(Atom)
                self.Charge.append(float(line[54:62]))
        PQRFile.close()
        self.AtomNum = len(self.Atoms)
        self.AtomPos = np.array(self.AtomPos)
        self.Charge = np.array(self.Charge, float)
        for idx, iPos in enumerate(self.AtomPos):
            self.Atoms[idx].position(iPos)
        return

    def get_pka_info(self):
        print('get pKa information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        pKaFile = open(self.filename+'.propka')
        self.pKaSite = 0.0
        self.pKa = []; self.pKaName = []
        self.pKaCt = 0.; self.PKaNt = 0.
        for line in pKaFile:
            if len(line)<24:
                continue
            if line[23]=='%':
                if line[0:2]=='C-': self.pKaCt = float(line[11:16])
                if line[0:2]=='N+': self.pKaNt = float(line[11:16])
                resid = int(line[3:7])
                if resid!=self.ResID:
                    self.pKa.append(float(line[11:16]))
                    self.pKaName.append(line[0:3])
                else:
                    self.pKaSite = float(line[11:16])
        pKaFile.close()
        return

    def construct_index_list(self, CutNear=10.):
        """ Lists that contains atom index
            first index: 0 mutsite, 1 other near, 2 all
            seconde index: 0 C, 1 N, 2 O, 3 S, 4 H, 5 heavy, 6 all
        """
        print('constructing index list >>>>>>>>>>>>>>>>>>>>>>>>>>>')
        heavy = ['C', 'N', 'O', 'S']
        IndexList = [[[] for i in range(7)] for i in range(3)]

        IndexMutSite = []; PosMutSite = []
        for idx, iAtom in enumerate(self.Atoms):
            if iAtom.ResID==self.ResID:
                IndexMutSite.append(idx)
                PosMutSite.append(iAtom.pos)

        PosMutSite = np.array(PosMutSite)
        # collect Near Residue ID. Not just the near atoms, but the atoms of near residue
        NearRes = []
        for idx, iAtom in enumerate(self.Atoms):
            ChainResID = iAtom.Chain+str(iAtom.ResID)
            if ChainResID not in NearRes and ChainResID != self.Chain+str(self.ResID):
                if np.min(np.linalg.norm(iAtom.pos-PosMutSite, axis=1)) < CutNear:
                    NearRes.append(ChainResID)
        NearRes = []
        for idx, iAtom in enumerate(self.Atoms):
            dis = 100000.0
            for j in IndexMutSite:
                tmpdis = np.linalg.norm(iAtom.pos-self.Atoms[j].pos)
                if tmpdis < dis:
                    dis = tmpdis
            if dis<CutNear and iAtom.ResID not in NearRes and iAtom.ResID!=self.ResID:
                NearRes.append(iAtom.ResID)
        NearAtom = []
        for idx, iAtom in enumerate(self.Atoms):
            if iAtom.Chain+str(iAtom.ResID) in NearRes:
                NearAtom.append(idx)
        # Index of mutation site
        for idx in IndexMutSite:
            IndexList[0][ele2index[self.Atoms[idx].atype]].append(idx)
            if self.Atoms[idx].atype in heavy:
                IndexList[0][5].append(idx)
            IndexList[0][6].append(idx)
        # Index of near atoms
        for idx in NearAtom:
            IndexList[1][ele2index[self.Atoms[idx].atype]].append(idx)
            if self.Atoms[idx].atype in heavy:
                IndexList[1][5].append(idx)
            IndexList[1][6].append(idx)
        # Index of all atoms
        for idx, iAtom in enumerate(self.Atoms):
            IndexList[2][ele2index[self.Atoms[idx].atype]].append(idx)
            if iAtom.atype in heavy:
                IndexList[2][5].append(idx)
            IndexList[2][6].append(idx)

        for i in range(3):
            for j in range(7):
                IndexList[i][j]=np.array(IndexList[i][j], int)
        return IndexList # construct_index_list()

    def setup_pairwise_interaction(self, sCut=10., lCut=40, FRI=FRIDefault):
        print('setup pairwise interaction >>>>>>>>>>>>>>>>>>>>>>>>')
        self.CLB = np.zeros([self.AtomNum, 5], float)
        self.VDW = np.zeros([self.AtomNum, 5], float)
        self.RIG = np.zeros([self.AtomNum, len(FRI), 5], float)
        t = cKDTree(self.AtomPos)
        NbShort = cKDTree.query_pairs(t, sCut)
        NbLong  = cKDTree.query_pairs(t, lCut)
        # Short cutoff for VDW
        for (i, j) in NbShort:
            ei    = self.Atoms[i].atype.replace(' ', '')
            ej    = self.Atoms[j].atype.replace(' ', '')
            dis   = np.linalg.norm(self.AtomPos[i]-self.AtomPos[j])
            ratio = (self.Atoms[i].R+self.Atoms[j].R)*(self.Atoms[i].R+self.Atoms[j].R)/dis
            vdw   = np.power(ratio, 12) - 2.*np.power(ratio, 6)
            self.VDW[i, ele2index[ej]] += vdw
            self.VDW[j, ele2index[ei]] += vdw
            #print(vdw)
        # Long cutoff for CLB and RIG
        for (i, j) in NbLong:
            ei  = self.Atoms[i].atype.replace(' ', '')
            ej  = self.Atoms[j].atype.replace(' ', '')
            dis = np.linalg.norm(self.AtomPos[i]-self.AtomPos[j])

            clb = self.Atoms[i].Charge*self.Atoms[j].Charge/dis
            self.CLB[i, ele2index[ej]] += clb
            self.CLB[j, ele2index[ei]] += clb
            #print(clb)

    def rips_complex(self, cutoff=16, deathcut=6):
        elecomb = ['C', 'N', 'O']
        Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
        def BinID(x, B):
            for i in range(len(B)-1):
                if B[i] <= x <= B[i+1]:
                    y = i
            return y

        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        self.rips_dth = np.zeros([3,3,12], int)
        self.rips_bar = np.zeros([3,3,12], int)
        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                matrixA = np.ones((len(pts), len(pts)))*100.
                for ii in range(len(pts)):
                    matrixA[ii, ii] = 0.
                    for jj in range(ii+1, len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dis = np.linalg.norm(pts[ii]-pts[jj])
                            matrixA[ii, jj] = dis
                            matrixA[jj, ii] = dis
        
                rips_complex = gudhi.RipsComplex(distance_matrix=matrixA, max_edge_length=deathcut)
                PH = rips_complex.create_simplex_tree().persistence()

                tmpbars = np.zeros(len(pts), dtype=dt)
                cnt = 0
                for simplex in PH:
                    dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                    if d-b < 0.1: continue
                    tmpbars[cnt]['dim']   = dim
                    tmpbars[cnt]['birth'] = b
                    tmpbars[cnt]['death'] = d
                    cnt += 1
                bars = tmpbars[0:cnt]
                for bar in bars:
                    death = bar['death']
                    if death >= deathcut: continue
                    Did = BinID(death, Bins)
                    self.rips_dth[Ip, Is,  Did] += 1
                    self.rips_bar[Ip, Is, :Did] += 1

    def rips_complex_spectra(self, cutoff=16):
        elecomb = ['C', 'N', 'O']
        Bins = [0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0]
        features = np.zeros((len(Bins)*len(elecomb)*len(elecomb), 8), np.float)
        def BinID(x, B):
            for i in range(len(B)-1):
                if B[i] <= x <= B[i+1]:
                    y = i
            return y

        Atoms = self.Atoms
        for iAtom in Atoms:
            if iAtom.verboseType.replace(' ', '')=="CA" and iAtom.ResID==self.ResID:
                AtomCA = iAtom
        res_num = self.ResID

        for Ip, ele1 in enumerate(elecomb):
            for Is, ele2 in enumerate(elecomb):
                pts = []; ptid = [] # ptid to indicate if atom belongs to mutated residue id
                for iAtom in Atoms:
                    if (iAtom.atype.replace(' ','')==ele1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==ele2 and iAtom.ResID==res_num):
                        dis = np.linalg.norm(iAtom.pos-AtomCA.pos)
                        if dis <= cutoff:
                            pts.append(iAtom.pos)
                            if iAtom.ResID==res_num:
                                ptid.append(0)
                            else:
                                ptid.append(1)
                matrixA = np.ones((len(pts), len(pts)))*100.
                for ii in range(len(pts)):
                    matrixA[ii, ii] = 0.
                    for jj in range(ii+1, len(pts)):
                        if ptid[ii]+ptid[jj]==1:
                            dis = np.linalg.norm(pts[ii]-pts[jj])
                            matrixA[ii, jj] = dis
                            matrixA[jj, ii] = dis

                for idx_cut, cut in enumerate(Bins):
                    Laplacian = np.zeros((len(pts), len(pts)), np.int)
                    Laplacian[matrixA < cut] = -1
                    Laplacian += np.diagflat(-np.sum(Laplacian, axis=0))
                    eigens = np.sort(np.linalg.eigvalsh(Laplacian))

                    idx_feat = idx_cut * 3 * 3 + Ip * 3 + Is
                    eigens = eigens[eigens > 10 ** -8]
                    if len(eigens) > 0:
                        # sum, min, max, mean, std, var,
                        features[idx_feat][0] = eigens.sum()
                        features[idx_feat][1] = eigens.min()
                        features[idx_feat][2] = eigens.max()
                        features[idx_feat][3] = eigens.mean()
                        features[idx_feat][4] = eigens.std()
                        features[idx_feat][5] = eigens.var()
                        features[idx_feat][6] = np.dot(eigens, eigens)
                        features[idx_feat][7] = len(eigens[eigens > 10 ** -8])

        return features.flatten()


    def alpha_complex(self):
        ElementList = ['C', 'N', 'O']
        res_num = self.ResID

        self.alpha_PH12 = np.zeros([3, 3, 14])
        dt = np.dtype([('dim', int), ('birth', float), ('death', float)])

        for Ip, e1 in enumerate(ElementList):
            for Is, e2 in enumerate(ElementList):
                points = []
                for iAtom in self.Atoms:
                    if (iAtom.atype.replace(' ','')==e1 and iAtom.ResID!=res_num) or \
                       (iAtom.atype.replace(' ','')==e2 and iAtom.ResID==res_num):
                        points.append(iAtom.pos)
                alpha_complex = gudhi.AlphaComplex(points=points)
                PH = alpha_complex.create_simplex_tree().persistence()

                tmpbars = np.zeros(len(PH), dtype=dt)
                cnt = 0
                for simplex in PH:
                    dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
                    if d-b < 0.1: continue
                    tmpbars[cnt]['dim']   = dim
                    tmpbars[cnt]['birth'] = b
                    tmpbars[cnt]['death'] = d
                    cnt += 1
                bars = tmpbars[0:cnt]
                if len(bars[bars['dim']==1]['death']) > 0:
                    self.alpha_PH12[Ip, Is, 0] = np.sum(bars[bars['dim']==1]['death'] - \
                                                        bars[bars['dim']==1]['birth'])
                    self.alpha_PH12[Ip, Is, 1] = np.max(bars[bars['dim']==1]['death'] - \
                                                        bars[bars['dim']==1]['birth'])
                    self.alpha_PH12[Ip, Is, 2] = np.mean(bars[bars['dim']==1]['death'] - \
                                                         bars[bars['dim']==1]['birth'])
                    self.alpha_PH12[Ip, Is, 3] = np.min(bars[bars['dim']==1]['birth'])
                    self.alpha_PH12[Ip, Is, 4] = np.max(bars[bars['dim']==1]['birth'])
                    self.alpha_PH12[Ip, Is, 5] = np.min(bars[bars['dim']==1]['death'])
                    self.alpha_PH12[Ip, Is, 6] = np.max(bars[bars['dim']==1]['death'])
                if len(bars[bars['dim']==2]['death']) > 0:
                    self.alpha_PH12[Ip, Is, 7]  = np.sum(bars[bars['dim']==2]['death'] - \
                                                         bars[bars['dim']==2]['birth'])
                    self.alpha_PH12[Ip, Is, 8]  = np.max(bars[bars['dim']==2]['death'] - \
                                                         bars[bars['dim']==2]['birth'])
                    self.alpha_PH12[Ip, Is, 9]  = np.mean(bars[bars['dim']==2]['death'] - \
                                                          bars[bars['dim']==2]['birth'])
                    self.alpha_PH12[Ip, Is, 10] = np.min(bars[bars['dim']==2]['birth'])
                    self.alpha_PH12[Ip, Is, 11] = np.max(bars[bars['dim']==2]['birth'])
                    self.alpha_PH12[Ip, Is, 12] = np.min(bars[bars['dim']==2]['death'])
                    self.alpha_PH12[Ip, Is, 13] = np.max(bars[bars['dim']==2]['death'])

        self.alpha_PH12_all = np.zeros([14])
        points = []
        for iAtom in self.Atoms:
            if iAtom.atype.replace(' ', '') != 'H':
                points.append(iAtom.pos)
        alpha_complex = gudhi.AlphaComplex(points=points)
        PH = alpha_complex.create_simplex_tree().persistence()

        tmpbars = np.zeros(len(PH), dtype=dt)
        cnt = 0
        for simplex in PH:
            dim, b, d = int(simplex[0]), float(simplex[1][0]), float(simplex[1][1])
            if d-b < 0.1: continue
            tmpbars[cnt]['dim']   = dim
            tmpbars[cnt]['birth'] = b
            tmpbars[cnt]['death'] = d
            cnt += 1
        bars = tmpbars[0:cnt]
        if len(bars[bars['dim']==1]['death']) > 0:
            self.alpha_PH12_all[0] = np.sum(bars[bars['dim']==1]['death'] - \
                                            bars[bars['dim']==1]['birth'])
            self.alpha_PH12_all[1] = np.max(bars[bars['dim']==1]['death'] - \
                                            bars[bars['dim']==1]['birth'])
            self.alpha_PH12_all[2] = np.mean(bars[bars['dim']==1]['death'] - \
                                             bars[bars['dim']==1]['birth'])
            self.alpha_PH12_all[3] = np.min(bars[bars['dim']==1]['birth'])
            self.alpha_PH12_all[4] = np.max(bars[bars['dim']==1]['birth'])
            self.alpha_PH12_all[5] = np.min(bars[bars['dim']==1]['death'])
            self.alpha_PH12_all[6] = np.max(bars[bars['dim']==1]['death'])
        if len(bars[bars['dim']==2]['death']) > 0:
            self.alpha_PH12_all[7]  = np.sum(bars[bars['dim']==2]['death'] - \
                                             bars[bars['dim']==2]['birth'])
            self.alpha_PH12_all[8]  = np.max(bars[bars['dim']==2]['death'] - \
                                             bars[bars['dim']==2]['birth'])
            self.alpha_PH12_all[9]  = np.mean(bars[bars['dim']==2]['death'] - \
                                              bars[bars['dim']==2]['birth'])
            self.alpha_PH12_all[10] = np.min(bars[bars['dim']==2]['birth'])
            self.alpha_PH12_all[11] = np.max(bars[bars['dim']==2]['birth'])
            self.alpha_PH12_all[12] = np.min(bars[bars['dim']==2]['death'])
            self.alpha_PH12_all[13] = np.max(bars[bars['dim']==2]['death'])
        

    def construct_feature_global(self):
        print('construct features >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        IndexArray = [np.array([0], int),
                      np.array([1], int),
                      np.array([2], int),
                      np.array([3], int),
                      np.array([4], int),
                      np.array([0,1,2,3], int),
                      np.array([0,1,2,3,4], int)]
        FeatureGLB = []
        # Charge
        for i in range(3):
            for j in range(7):
                FeatureGLB.append(np.sum(self.Charge[self.IndexList[i][j]]))
                FeatureGLB.append(np.sum(np.abs(self.Charge[self.IndexList[i][j]])))
        # RIG
        #for i in range(3):
            #for j in [0,1,2,3,5]:
                #FeatureGLB.append(np.sum(self.RIG[self.IndexList[i][j],:][:,IndexArray[5]]))
        # VDW
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.VDW[self.IndexList[i][j],:][:,IndexArray[5]]))
        # CLB
        for i in range(3):
            for j in [0,1,2,3,5]:
                FeatureGLB.append(np.sum(self.CLB[self.IndexList[i][j]][:,IndexArray[6]]))
                FeatureGLB.append(np.sum(np.abs(self.CLB[self.IndexList[i][j]][:,IndexArray[6]])))
        self.FeatureGLB = FeatureGLB
        FeatureGLBother = []
        # Other
        AA = self.ResName
        for Group in Groups:
            if AA in Group:
                FeatureGLBother.append(1.0)
            else:
                FeatureGLBother.append(0.0)
        FeatureGLBother.append(AAvolume[AA])
        FeatureGLBother.append(AAhydropathy[AA])
        FeatureGLBother.append(AAarea[AA])
        FeatureGLBother.append(AAweight[AA])
        FeatureGLBother.append(AAcharge(AA))
        FeatureGLBother.extend(AApharma[AA])
        self.FeatureGLBother = FeatureGLBother
    
    def construct_feature_env(self):
        print('construct environment feature >>>>>>>>>>>>>>>>>>>>>')
        FeatureEnv = []
        NearSeq = []
        CurResID = -1000
        for i in self.IndexList[1][6]:
            ResID = self.Atoms[i].ResID
            if self.Atoms[i].ResID!=CurResID:
                CurResID = ResID
                NearSeq.append(self.Atoms[i].ResName)
        for Group in Groups:
            cnt = 0.
            for AA in NearSeq:
                if AA in Group:
                    cnt += 1.
            FeatureEnv.append(cnt)
            FeatureEnv.append(cnt/max(1., float(len(NearSeq))))
        Vol = []; Hyd = []; Area = []; Wgt = []; Chg = []
        phara = [0, 0, 0, 0, 0, 0]
        for AA in NearSeq:
            Vol.append(AAvolume[AA])
            Hyd.append(AAhydropathy[AA])
            Area.append(AAarea[AA])
            Wgt.append(AAweight[AA])
            Chg.append(AAcharge(AA))
            for i in range(6):
                phara[i] += AApharma[AA][i]
        Vol = np.asarray(Vol)
        Hyd = np.asarray(Hyd)
        Area = np.asarray(Area)
        Wgt = np.asarray(Wgt)

        if len(NearSeq) == 0:
            FeatureEnv.extend([0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.])
        else:
            FeatureEnv.extend([np.sum(Vol), np.sum(Vol)/float(len(NearSeq)), np.var(Vol)])
            FeatureEnv.extend([np.sum(Hyd), np.sum(Hyd)/float(len(NearSeq)), np.var(Hyd)])
            FeatureEnv.extend([np.sum(Area), np.sum(Area)/float(len(NearSeq)), np.var(Area)])
            FeatureEnv.extend([np.sum(Wgt), np.sum(Wgt)/float(len(NearSeq)), np.var(Wgt)])
        FeatureEnv.append(sum(Chg))
        FeatureEnv.extend(phara)

        self.FeatureEnv = FeatureEnv

    def construct_feature_MIBPB(self, h=0.5):
        Area = []
        SolvEng = []
        print('run MIBPB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        if not os.path.exists('./'+self.filename+'.englist') or \
           not os.path.exists('./'+self.filename+'.eng') or \
           not os.path.exists('./'+self.filename+'.arealist') or \
           not os.path.exists('./'+self.filename+'.areavolume'):
            #print('./mibpb5 '+self.filename+' h=%f'%(h))
            os.system('mibpb5 '+self.filename+' h=%f'%(h))
            os.system('mv partition_area.txt '+self.filename+'.arealist')
            os.system('mv area_volume.dat '+self.filename+'.areavolume')
        os.system('rm -f bounding_box.txt')
        os.system('rm -f grid_info.txt')
        os.system('rm -f intersection_info.txt')
        os.system('rm -f '+self.filename+'.dx')
        # Info from arealist file
        AreaListFile = open(self.filename+'.arealist')
        for idx, line in enumerate(AreaListFile):
            a, b = line.split()
            Area.append(float(b))
        AreaListFile.close()
        # Info from englist file
        EngListFile = open(self.filename+'.englist')
        for idx, line in enumerate(EngListFile):
            SolvEng.append(float(line))
        EngListFile.close()
        Area = np.array(Area, float)
        SolvEng = np.array(SolvEng, float)

        # Info from areavolume file
        AreaVolumeFile = open(self.filename+'.areavolume')
        TotalArea = float(AreaVolumeFile.readline())
        TotalVolume = float(AreaVolumeFile.readline())
        AreaVolumeFile.close()
        # Info from eng file
        EngFile = open(self.filename+'.eng')
        EngFile.readline()
        TotalSolvEng = float(EngFile.readline())
        EngFile.close()

        FeatureMIBPB = []
        # SolvEng from MIBPB
        for i in range(3):
            for j in range(7):
                FeatureMIBPB.append(np.sum(SolvEng[self.IndexList[i][j]]))

        print(Area)
        # Area from MIBPB
        for i in range(3):
            for j in range(7):
                FeatureMIBPB.append(np.sum(Area[self.IndexList[i][j]]))
        self.FeatureMIBPB = FeatureMIBPB

        FeatureMIBPBglb = []
        # Global
        FeatureMIBPBglb.append(TotalSolvEng)
        FeatureMIBPBglb.append(TotalArea)
        FeatureMIBPBglb.append(TotalVolume)
        self.FeatureMIBPBglb = FeatureMIBPBglb

    def runBLAST(self):
        print('run BLAST >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        cline = NcbipsiblastCommandline(query=self.filename_single+'.fasta',
                                        db='../../../S2648/bin/blastdb/nr/nr_db',
                                        num_iterations=3,
                                        evalue=5000,
                                        out=self.filename_single+'.out',
                                        out_ascii_pssm=self.filename_single+'.pssm')

        if not os.path.exists(self.filename_single+'.pssm'):
            print('running '+self.filename_single+'.pssm')
            stdout, stderr = cline()
            print(stdout, stderr)
        else:
            flag = True
            fp = open(self.filename_single+'.pssm')
            for line in fp:
                if line[:10]=='PSI Gapped':
                    flag = False
            fp.close()
            if flag:
                print('running '+self.filename_single+'.pssm')
                stdout, stderr = cline()
                print(stdout, stderr)
        return # runBLAST

    def construct_feature_seq(self):
        print('generate secondary structure information >>>>>>>>>>')
        FeatureSeq = []

        # Structure based DSSP
        parser = PDBParser()
        structure = parser.get_structure(self.PDBid, self.filename+'.pdb')
        model = structure[0]
        dssp = DSSP(model, self.filename+'.pdb', dssp='mkdssp')
        ssindex = ss2index[dssp[(self.Chain, (' ', self.ResID, ' '))][2]]
        FeatureSeq.append(ssindex)

        # Sequence based Spider3
        #print(os.path.exists('../../bin/SPIDER2_local/misc/pred_pssm.py'))
        if not os.path.exists('../../bin/SPIDER2_local/misc/pred_pssm.py'):
            sys.exit('Please make sure the SPIDER2_local exists')
        os.system('../../bin/SPIDER2_local/misc/pred_pssm.py '+self.filename_single+'.pssm -f')
        spdfile = open(self.filename_single+'.spd3')
        #spdfile.readline()
        #for line in spdfile:
        #    if int(line.split()[0]) == self.ResIDSeq:
        #        break
        lines = spdfile.read().splitlines()
        line = lines[self.ResIDSeq+1] # Here, num+1 is because of the first line of spd3 is header
        #if line.split()[1] != self.ResName:
            #print(self.ResIDSeq, line.split()[1], self.ResName)
            #print(self.filename_single+'.fasta is removed')
            #print(self.filename_single+'.pssm is removed')
            #os.system('rm '+self.filename_single+'.fasta')
            #os.system('rm '+self.filename_single+'.pssm')
            #sys.exit('Wrong residue when calling pssm for '+self.typeFlag)
        psi=0.; phi=0.; pc=0.; pe=0.; ph=0.
        d0, d1, d2, d3, phi, psi, d4, d5, pc, pe, ph = line.split()
        spdfile.close()
        FeatureSeq.extend([float(phi), float(psi), float(pc), float(pe), float(ph)])
        self.FeatureSeq = FeatureSeq

def construct_feature_PH(p_WT, p_MT):
    # rips complex
    Feature_0_dth = np.zeros([324], float)
    Feature_0_bar = np.zeros([324], float)

    Feature_i_dth = p_MT.rips_dth-p_WT.rips_dth
    Feature_i_bar = p_MT.rips_bar-p_WT.rips_bar
    for i0 in range(3):
        for i1 in range(3):
            for i2 in range(12):
                idx = i0*108 + i1*36 + i2*3
                Feature_0_dth[idx]   = p_WT.rips_dth[i0, i1, i2]
                Feature_0_bar[idx]   = p_WT.rips_bar[i0, i1, i2]
                Feature_0_dth[idx+1] = p_MT.rips_dth[i0, i1, i2]
                Feature_0_bar[idx+1] = p_MT.rips_bar[i0, i1, i2]
                Feature_0_dth[idx+2] = Feature_i_dth[i0, i1, i2]
                Feature_0_bar[idx+2] = Feature_i_bar[i0, i1, i2]

    return np.array(Feature_0_dth)

def construct_feature_PH12(p_WT, p_MT):
    # alpha complex
    Feature_1_2 = np.zeros([378], float)

    Feature_i_1_2 = p_MT.alpha_PH12-p_WT.alpha_PH12
    for i0 in range(3):
        for i1 in range(3):
            for i2 in range(14):
                idx = ((i0*3+i1)*14+i2)*3
                Feature_1_2[idx]   = p_WT.alpha_PH12[i0, i1, i2]
                Feature_1_2[idx+1] = p_MT.alpha_PH12[i0, i1, i2]
                Feature_1_2[idx+2] = Feature_i_1_2[i0, i1, i2]

    Feature_1_2_all = np.zeros([42], float)
    Feature_1_2_all[ 0:14] = p_WT.alpha_PH12_all
    Feature_1_2_all[14:28] = p_MT.alpha_PH12_all
    Feature_1_2_all[28:42] = p_MT.alpha_PH12_all-p_WT.alpha_PH12_all

    #Feature = np.concatenate((Feature_0_dth, Feature_1_2, Feature_1_2_all), axis=0)
    Feature = np.concatenate((Feature_1_2, Feature_1_2_all), axis=0)
    #Feature = np.concatenate((Feature_0_dth, Feature_1_2), axis=0)

    return np.array(Feature)
 
def construct_feature_aux(p_WT, p_MT, flag_MIBPB=False, flag_BLAST=False):
    if flag_MIBPB:
        p_MTFeature = p_MT.FeatureMIBPB+p_MT.FeatureGLB+p_MT.FeatureMIBPBglb+p_MT.FeatureGLBother
        p_WTFeature = p_WT.FeatureMIBPB+p_WT.FeatureGLB+p_WT.FeatureMIBPBglb+p_WT.FeatureGLBother
    else:
        p_MTFeature = p_MT.FeatureGLB+p_MT.FeatureGLBother
        p_WTFeature = p_WT.FeatureGLB+p_WT.FeatureGLBother
    Feature = p_MTFeature+p_WTFeature
    Feature.extend(map(operator.sub, p_MTFeature, p_WTFeature))

    # pKa features
    pKaIndex = {'ASP':0, 'GLU':1, 'ARG':2, 'LYS':3, 'HIS':4, 'CYS':5, 'TYR':6}
    pKaGroup = ['ASP',   'GLU',   'ARG',   'LYS',   'HIS',   'CYS',   'TYR']
    mutpKa   = np.array(p_MT.pKa, float)
    wildpKa  = np.array(p_WT.pKa, float)
    wildpKaname = p_WT.pKaName
    defer = mutpKa-wildpKa
    absmax = np.max(np.abs(defer))
    abssum = np.sum(np.abs(defer))
    maxpos = np.max(defer)
    maxneg = np.min(defer)
    netchange = np.sum(defer)
    DetailShiftAbs = np.zeros([7], float)
    DetailShiftNet = np.zeros([7], float)
    for j in range(len(wildpKa)):
        if wildpKaname[j] in pKaGroup:
            DetailShiftAbs[pKaIndex[wildpKaname[j]]] += np.abs(mutpKa[j]-wildpKa[j])
            DetailShiftNet[pKaIndex[wildpKaname[j]]] += mutpKa[j]-wildpKa[j]
    mutC = p_MT.pKaCt; mutN = p_MT.pKaNt
    wildC = p_WT.pKaCt; wildN = p_WT.pKaNt
    mutsitepKa = p_MT.pKaSite; wildsitepKa = p_WT.pKaSite;
    Feature.extend([absmax, abssum, maxpos, maxneg, netchange, 
                    wildsitepKa, mutsitepKa, mutsitepKa-wildsitepKa, 
                    wildC, mutC, mutC-wildC, wildN, mutN, mutN-wildN])
    Feature.extend(DetailShiftNet.tolist())
    Feature.extend(DetailShiftAbs.tolist())

    # Environment features
    Feature.extend(p_WT.FeatureEnv)

    if flag_BLAST:
        # PSSM features
        AAind = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10, \
                 'L':11,'K':12,'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,'Y':19,'V':20}
        resWT = p_WT.ResName
        resMT = p_MT.ResName
        resNum = p_MT.ResIDSeq
        pssm_score1 = np.zeros([p_WT.SeqLength, 20])
        pssm_score2 = np.zeros([p_WT.SeqLength, 20])
        pssm_score3 = np.zeros([p_WT.SeqLength, 2])

        pssmfile = open(p_WT.filename_single+'.pssm')
        lines = pssmfile.read().splitlines()
        for idx, line in enumerate(lines[3:3+p_WT.SeqLength]):
            tmp_vec = line.split()
            pssm_score1[idx, :] = tmp_vec[2:22]
            pssm_score2[idx, :] = tmp_vec[22:42]
            pssm_score3[idx, :] = tmp_vec[42:]
        pssmfile.close()

        Feature.append(pssm_score1[p_MT.ResIDSeq, AAind[resMT]-1])
        Feature.append(pssm_score1[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(pssm_score1[p_MT.ResIDSeq, AAind[resMT]-1] \
                      -pssm_score1[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(np.sum(pssm_score1[p_WT.ResIDSeq, :]))

        Feature.append(pssm_score2[p_MT.ResIDSeq, AAind[resMT]-1])
        Feature.append(pssm_score2[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(pssm_score2[p_MT.ResIDSeq, AAind[resMT]-1] \
                      -pssm_score2[p_WT.ResIDSeq, AAind[resWT]-1])
        Feature.append(np.sum(pssm_score2[p_WT.ResIDSeq, :]))

        Feature.extend(pssm_score3[p_WT.ResIDSeq].tolist())

        # SS features
        Feature.extend(p_MT.FeatureSeq)
        Feature.extend(p_WT.FeatureSeq)
        Feature.extend(map(operator.sub, p_MT.FeatureSeq, p_WT.FeatureSeq))
    #print(len(Feature))

    return np.array(Feature) # construct_feature_aux