import os, re, sys, warnings, time
import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.PDBIO import PDBIO
import multiprocessing as mp
import pandas as pd 
import glob 
from mutsol_protein import Protein, construct_feature_aux, construct_feature_PH, construct_feature_PH12

AminoA = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU', 'SER', 'THR', 'ASN', 'GLN', 'CYS',
          'SEF', 'GLY', 'PRO', 'ALA', 'VAL', 'ILE', 'LEU', 'MET', 'PHE', 'TYR',
          'TRP']
# Non-canonical or non natural
NC_AminoA = {'LLP': 'LYS', 'M3P': 'LYS', 'MSE': 'MET', 'F2F': 'PHE', 'CGU': 'GLU',
        'MYL': 'LYS', 'TPO': 'THR', 'HSE': 'HIS'}
default_cutoff = 16.
ElementList = ['C', 'N', 'O']

def use_profix_scap(filename, muteChain, resID, resMT):
    # the profix eliminates the insertion code!!!
    # adjust atoms position by profix
    #if not os.path.exists('profix'):
    os.system('cp ../../bin/profix .')
    #os.system('cp ../../bin/jackal.dir .')
    os.system('./profix -fix 0 '+filename+'_WT.pdb')
    os.system('mv '+filename+'_WT_fix.pdb '+filename+'_WT.pdb')
 
    # generate mutant PDB_file
    os.system('cp ../../bin/scap .')
    scap_file = open('tmp_scap.list', 'w')
    scap_file.write(','.join([muteChain, str(resID), resMT]))
    scap_file.close()
    os.system('./scap -ini 20 -min 4 '+filename+'_WT.pdb tmp_scap.list')
    os.system('mv '+filename+'_WT_scap.pdb '+filename+'_MT.pdb')
    os.system('rm -f tmp_scap.list')
    return

def removeChains_insertionCode(structure, Chains, resID, muteChain):
    '''
        This function exclude chains that are not in 'Chains'
        structure: the structure from PDBs.pdb
        Chains: target chains are included
        resID: mutation residue ID
        muteChain: mutation chain
    '''
    # the write-down PDBs are for MIBPB calculation
    # the SolvEng has a (1.421162, 0.704903) might be important with more juice
    # however, insertion code should be removed since profix will remove it anyway
    # the new residue ID for mutation site should be updated
    structure_clean = structure.copy()
    ichain_ids_to_remove = []
    for ichain in structure_clean[0]:
        #print(ichain.id)
        if ichain.id not in Chains:
            ichain_ids_to_remove.append(ichain.id)
        else: # remove the insertion code
    # when change residue ID with insertion code
    # ValueError: Cannot change id from `(' ', 83, ' ')` to `(' ', 86, ' ')`. 
    # The id `(' ', 86, ' ')` is already used for a sibling of this entity.
            for idx, iresidue in enumerate(structure_clean[0][ichain.id]):
                #print(idx, iresidue.id, resID, muteChain, ichain.id)
                if iresidue.id == resID and ichain.id == muteChain:
                    resID_structure = idx+1
                    print(resID_structure)
                iresidue_id = list(iresidue.id)
                iresidue_id[0] = 'Old'
                iresidue.id = tuple(iresidue_id)
            for idx, iresidue in enumerate(structure_clean[0][ichain.id]):
                iresidue_id = list(iresidue.id)
                iresidue_id[0], iresidue_id[1], iresidue_id[2] = ' ', idx+1, ' '
                iresidue.id = tuple(iresidue_id)

    #print(ichain_ids_to_remove)          
    for ichain_id in ichain_ids_to_remove:
        structure_clean[0].detach_child(ichain_id)
    return structure_clean, resID_structure


class get_structure: 
    def __init__(self, PDBid, Chains, muteChain, resWT, resID, resMT, pH='7.0', cutoff=default_cutoff, onlyBLAST=False):
        self.PDBid = PDBid 

        self.pH = str(pH)
        self.cutoff = default_cutoff
        
        self.muteChain = muteChain
        self.resWT = resWT
        self.resMT = resMT
        self.fasta = {}
        self.Chains = Chains

        # filename
        self.fileComplex  = PDBid+'_'+Chains

        # deal with insertion code of PDB
        resIDchr = re.search('[a-zA-Z]', resID)
        if resIDchr!=None:
            resIDchr = resIDchr.group(0).upper()
            resIDidx = int(resID[:-1])
        else:
            resIDchr = ' '
            resIDidx = int(resID)
        self.resID_ori = (' ', resIDidx, resIDchr)

        # get PDB_File and fasta_File
        os.system('cp ../../pdb/'+self.PDBid+'.pdb ./'+self.PDBid+'.pdb')

        if not onlyBLAST:
            # change self.resID_ori (self.resID) to number residue
            # use Biopython to load PDB_file for the starting residue ID of target chain
            parser = PDBParser(PERMISSIVE=1)
            s = parser.get_structure(self.PDBid, self.PDBid+'.pdb')
            #print(list(s[0]['I']))
            # three things are done in the following
            # 1. remove ions and waters; 2. replace non-canonical residues to their parental residues
            for ichain in self.Chains:
                iresidue_ids_to_remove = []
                for iresidue in s[0][ichain]:
                    if iresidue.resname in NC_AminoA:
                        iresidue_id = list(iresidue.id)
                        iresidue_id[0] = ' '
                        iresidue.id = tuple(iresidue_id)
                        iresidue.resname = NC_AminoA[iresidue.resname]
                    elif iresidue.resname not in NC_AminoA and iresidue.resname not in AminoA:
                        iresidue_ids_to_remove.append(iresidue.id)
                        
                for iresidue_id in iresidue_ids_to_remove:
                    s[0][ichain].detach_child(iresidue_id)
            # 3. remove other chains
            ichain_ids_to_remove = []
            for ichain in s[0]:
                #print(s[0].id)
                if ichain.id not in self.Chains:
                    ichain_ids_to_remove.append(ichain.id)

            #print(ichain_ids_to_remove)
            for ichain_id in ichain_ids_to_remove:
                s[0].detach_child(ichain_id)

            #print(self.resID_ori)
            # save files and profix
            s, resID_temp = removeChains_insertionCode(s, self.Chains, self.resID_ori, self.muteChain)
            if not os.path.exists(f'{self.fileComplex}.pdb'):
                io_Complex = PDBIO()
                io_Complex.set_structure(s)
                io_Complex.save(self.fileComplex+'.pdb')
                os.system(f'profix -fix 0 {self.fileComplex}.pdb')
                os.system(f'mv {self.fileComplex}_fix.pdb {self.fileComplex}.pdb')
            parser = PDBParser(PERMISSIVE=1)
            self.s = parser.get_structure(self.PDBid, f'{self.fileComplex}.pdb')
            self.resID = (' ', resID_temp, ' ')

            # distance_mutation_binding
            for iresidue in self.s[0][muteChain]:
                if iresidue.id == self.resID:
                    #print(resID_temp, self.resWT, iresidue.id)
                    if not three_to_one(iresidue.resname) == self.resWT:
                        sys.exit('After first profix, mutant residue not match')
                    self.muteResidue = iresidue.copy()
            self.distance_muteResidue_bindingSite = 100
    
    def generateMutedPDBs(self, flag_use_binary = True):
        self.s_MutedPartner_WT, self.resID_MutedPartner = \
                removeChains_insertionCode(self.s, self.Chains, self.resID, self.muteChain)

        if flag_use_binary and (not os.path.exists(self.PDBid+'_WT.pdb') \
                            or  not os.path.exists(self.PDBid+'_MT.pdb')):
            io_MutedPartner_WT = PDBIO()
            io_MutedPartner_WT.set_structure(self.s_MutedPartner_WT)
            io_MutedPartner_WT.save(self.PDBid+'_WT.pdb')
            use_profix_scap(self.PDBid, self.muteChain, self.resID_MutedPartner, self.resMT)
        print('generate files:', self.PDBid+'_WT.pdb', self.PDBid+'_MT.pdb')

        parser = PDBParser(PERMISSIVE=1)
        self.s_MutedPartner_MT = parser.get_structure(self.PDBid, self.PDBid+'_MT.pdb')
        return # generateMutedPartnerPDBs

    def generateMutedPQRs(self):
        # generated PQR_file
        if not os.path.exists(self.PDBid+'_WT.pqr'):
            os.system('pdb2pqr --ff=amber --ph-calc-method=propka --chain --with-ph='+self.pH+
                    ' '+self.PDBid+'_WT.pdb '+self.PDBid+'_WT.pqr')
        if not os.path.exists(self.PDBid+'_MT.pqr'):
            os.system('pdb2pqr --ff=charmm --ph-calc-method=propka --chain --with-ph='+self.pH+
                    ' '+self.PDBid+'_MT.pdb '+self.PDBid+'_MT.pqr')
        return # generateMutedPartnerPQRs

    def readFASTA(self):
        # local structure only used for fasta. use original pdb, not the fixed one
        parser = PDBParser(PERMISSIVE=1)
        s = parser.get_structure(self.PDBid, self.PDBid+'.pdb')

        # initialize variables
        self.non_canonical = []

        # filename
        self.fileMuteChain = self.PDBid+'_'+self.muteChain

        # check missing residue and record it to AB_MISS_RES and AG_MISS_RES
        fp = open(self.PDBid+'.pdb')
        # this MISS_RES only record the target chain 'self.muteChain'
        array_RES = {}
        flagMISSRES = False; marker = ''
        for line in fp:
            words = re.split(' |\n', line)
            if len(words) > 4:
                if words[2] == 'MISSING' and words[3] == 'RESIDUES':
                    marker = words[1]
                    flagMISSRES = True
                    break
        if flagMISSRES:
            for _ in range(5): # skip 5 lines
                fp.readline()
            line = re.split(' |\n', fp.readline())
            line = [i for i in line if i]
            while line[1] == marker:
                if line[3] in self.muteChain:
                    if line[2] in AminoA:
                        array_RES[int(line[4])] = three_to_one(line[2])
                    #elif line[2] != 'HOH':
                    #    self.non_canonical.append(line[2])
                line = re.split(' |\n', fp.readline())
                line = [i for i in line if i]
        fp.close()

        # if array_RES is empty, then jump is not allowed or
        # the largest index of MISSING RESIDUE is less than starter residue
        residue = next(iter(s[0][self.muteChain])) # what's this? # gives the first residue
        start_idx_PDB  = residue.id[1]
        if len(array_RES) == 0:
            flagMISSRES = False
        else:
            array_RES_idx = list(array_RES.keys()) # find the last index
            last_idx_MISS = array_RES_idx[-1]
            if last_idx_MISS < start_idx_PDB:
                flagMISSRES = False
        # if flagMISSRES = False, then jump index is not allowed

        shift = 0;# last_idx 
        resID_mute_in_ = 0
        for idx, iresidue in enumerate(s[0][self.muteChain]):
            if iresidue.resname in AminoA:
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    if iresidue.id == self.resID_ori:
                        resID_mute_in_ = iresidue.id[1]+shift
                        if self.resWT != three_to_one(iresidue.resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 1')
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue.resname)
                else:
                    if iresidue.id == self.resID_ori:
                        # this might be OK for no MISSING RESIDUE, what about the other?
                        # should be OK, MISSING RESIDUE is in the front
                        resID_mute_in_ = start_idx_PDB+idx
                        if self.resWT != three_to_one(iresidue.resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 2')
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue.resname)
            elif iresidue.resname in NC_AminoA:
                iresidue_resname = NC_AminoA[iresidue.resname]
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    if iresidue.id == self.resID_ori:
                        resID_mute_in_ = iresidue.id[1]+shift
                        if self.resWT != three_to_one(iresidue_resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 3')
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue_resname)
                else:
                    if iresidue.id == self.resID_ori:
                        # this might be OK for no MISSING RESIDUE, what about the other?
                        # should be OK, MISSING RESIDUE is in the front
                        resID_mute_in_ = start_idx_PDB+idx
                        if self.resWT != three_to_one(iresidue_resname):
                            sys.exit('Check the PDB residues. Wrong residue name for input!!! 4')
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue_resname)
            #elif iresidue.resname != 'HOH' and start_idx_PDB+idx == iresidue.id[1]:
            #    self.non_canonical.append(iresidue.resname)
            #if iresidue.resname != 'HOH':
            #    print(start_idx_PDB+idx, iresidue.id, iresidue.resname)

        # sort array_RES by index
        array_RES_sorted = {}
        for idx in sorted(array_RES):
            array_RES_sorted[idx] = array_RES[idx]
        #print(array_RES_sorted.keys())

        # get the start and end index, real mutation ID
        array_RES_idx = list(array_RES_sorted.keys())
        idx_s = array_RES_idx[0]; idx_e = array_RES_idx[-1]
        # 0 in array_RES_idx, when flagMISSRES == True
        if 0 not in array_RES_idx and idx_s < 0:
            self.resID_fasta = resID_mute_in_ - idx_s - 1
            seqlength = idx_e-idx_s
        else:
            self.resID_fasta = resID_mute_in_ - idx_s
            seqlength = idx_e-idx_s + 1
        #print(self.resID_fasta, resID_mute_in_, idx_s, seqlength)

        flag_fastaWT_fastaMT = False
        #print(len(array_RES_idx))
        if seqlength == len(array_RES_idx):
            #print('This is a consecutive FASTA sequence!!!')
            self.fastaWT = ''.join(list(array_RES_sorted.values()))
            array_RES_sorted[resID_mute_in_] = self.resMT
            self.fastaMT = ''.join(list(array_RES_sorted.values()))
            flag_fastaWT_fastaMT = True

        #print('Cannot have a consecutive FASTA sequence from this PDB file')
        #print(idx_s, idx_e, idx_e-idx_s+1, len(array_RES_idx))
        #print('Use SEQRES info from PDB instead')

        if not flag_fastaWT_fastaMT:
            print('WARNING: flag_fastaWT_fastaMT ture!')
            # use SEQRES in PDB for fasta file
            array_RES = {}; idx = idx_s
            fp = open(self.PDBid+'.pdb')
            for line in fp:
                if line[:6] == 'SEQRES':
                    words = [i for i in re.split(' |\n', line) if i]
                    if words[2] == self.muteChain:
                        for resname in words[4:]:
                            if resname in AminoA:
                                array_RES[idx] = three_to_one(resname)
                            #elif resname != 'HOH':
                            #    self.non_canonical.append(resname)
                            idx += 1 # !!! 
            fp.close()
            if array_RES[resID_mute_in_] != self.resWT:
                # use the position info from BioPython to check whether it is the one of array_RES
                array_RES_index = list(array_RES_sorted.keys())
                resID_mute_in_pos = array_RES_index.index(resID_mute_in_)
                self.resID_fasta = resID_mute_in_pos # this is dangerous
                array_RES_residue = list(array_RES.values())
                if array_RES_residue[resID_mute_in_pos] == self.resWT:
                    self.fastaWT = ''.join(list(array_RES_residue))
                    array_RES_residue[resID_mute_in_pos] = self.resMT
                    self.fastaMT = ''.join(list(array_RES_residue))
                    flag_fastaWT_fastaMT = True
                else:
                    sys.exit('Need manually check or fasta file')
            else:
                self.fastaWT = ''.join(list(array_RES.values()))
                array_RES[resID_mute_in_] = self.resMT
                self.fastaMT = ''.join(list(array_RES.values()))
                flag_fastaWT_fastaMT = True

        if not flag_fastaWT_fastaMT:
            return False

        self.fasta['WT'] = self.fastaWT
        self.fasta['MT'] = self.fastaMT
        return True # readFASTA()

    def writeFASTA(self):
        seqlength = len(self.fasta['WT'])
        # check if self.fasta = {}, then run readFASTA
        if len(self.fasta) == 0:
            self.readFASTA()
        seqfile_WT = open(self.fileMuteChain+'_WT.fasta', 'w')
        seqfile_MT = open(self.fileMuteChain+'_MT.fasta', 'w')
        for idx in range(seqlength):
            seqfile_WT.write(self.fasta['WT'][idx])
            seqfile_MT.write(self.fasta['MT'][idx])
            if (idx+1)%80 == 0:
                seqfile_WT.write('\n')
                seqfile_MT.write('\n')
        seqfile_WT.close()
        seqfile_MT.close()
        if len(self.fasta['WT']) < 15:
            seqfile_WT = open(self.fileMuteChain+'_WT.fasta', 'w')
            seqfile_MT = open(self.fileMuteChain+'_MT.fasta', 'w')
            for idx in range(seqlength*20):
                seqfile_WT.write(self.fasta['WT'][idx%seqlength])
                seqfile_MT.write(self.fasta['MT'][idx%seqlength])
                if (idx+1)%80 == 0:
                    break
            seqfile_WT.close()
            seqfile_MT.close()

        return # writeFASTA()

    def writeFoldFASTA(self):

        if not os.path.exists('../../fasta/'):
            os.mkdir('../../fasta/')

        self.filefold = self.PDBid+'_'+self.muteChain + '_' + self.resWT + str(self.resID_ori[1]) + self.resMT

        seqlength = len(self.fasta['WT'])
        # check if self.fasta = {}, then run readFASTA
        if len(self.fasta) == 0:
            self.readFASTA()
        seqfile_WT = open('../../fasta/'+self.filefold+'_WT_AF.fasta', 'w')
        seqfile_WT.write('>'+self.PDBid+'_'+self.muteChain + '_WT\n')
        seqfile_MT = open('../../fasta/'+self.filefold+'_MT_AF.fasta', 'w')
        seqfile_MT.write('>'+self.filefold+'\n')
        for idx in range(seqlength):
            seqfile_WT.write(self.fasta['WT'][idx])

        seqlength = len(self.fasta['MT'])
        for idx in range(seqlength):
            seqfile_MT.write(self.fasta['MT'][idx])
        seqfile_WT.close()
        seqfile_MT.close()

        return # writeFASTA()

    def generateMutedFoldPDBs(self, debug=False):
        if not os.path.exists('../../alphafold/'):
            os.mkdir('../../alphafold/')
        
        self.filefold = self.PDBid+'_'+self.muteChain 
        jobname = self.filefold 
        if not os.path.exists('../../alphafold/'+jobname+'_WT') or len(list(glob.glob('../../alphafold/'+jobname+'_WT/*.pdb')))<5:
        #or open("../../alphafold/"+jobname+"_WT/log.txt", "r").readlines()[-1][-5:] != 'Done\n':
            if debug==False:
                os.system('colabfold_batch ../../fasta/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'_WT_AF.fasta ../../alphafold/'+jobname+'_WT')
            else:
                file = open("../../../debug.txt", "a+")
                file.write(jobname+"_"+self.resWT + str(self.resID_ori[1]) + self.resMT+"\n")
                file.close()
                return
        if not os.path.exists('../../alphafold/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'_MT') or len(list(glob.glob('../../alphafold/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'_MT/*.pdb')))<5:
        #or open("../../alphafold/"+jobname+ '_' + self.resWT + str(self.resID[1]) + self.resMT+"_MT/log.txt", "r").readlines()[-1][-5:] != 'Done\n':
            if debug==False:
                os.system('colabfold_batch ../../fasta/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'_MT_AF.fasta ../../alphafold/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'_MT')
            else:
                file = open("../../../debug.txt", "a+")
                file.write(jobname+"_"+self.resWT + str(self.resID_ori[1]) + self.resMT+"\n")
                file.close()

    def buildfold_dir(self):
        self.filefold = self.PDBid+'_'+self.muteChain 
        jobname = self.filefold 
        wild_name = '../../alphafold/'+jobname+'_WT'
        mut_name = '../../alphafold/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'_MT'

        for idx in ['001', '002', '003', '004', '005']:

            wild_files = list(glob.glob(wild_name+'/*_{}_*.pdb'.format(idx)))
            mut_files = list(glob.glob(mut_name+'/*_{}_*.pdb'.format(idx)))

            if not os.path.exists('../../feature_rank_{}/'.format(idx)+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT):
                if len(str(self.resID_ori[1])) == 1:
                    os.mkdir('../../feature_rank_{}/'.format(idx)+jobname+ '_' + self.resWT + '0' + str(self.resID_ori[1]) + self.resMT)
                else:
                    os.mkdir('../../feature_rank_{}/'.format(idx)+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT)
            if len(str(self.resID_ori[1])) == 1:
                os.system('cp '+wild_files[0]+' ../../feature_rank_{}/'.format(idx)+'/'+jobname+ '_' + self.resWT + '0' + str(self.resID_ori[1]) + self.resMT+'/'+self.PDBid+'_WT.pdb')
                os.system('cp '+mut_files[0]+' ../../feature_rank_{}/'.format(idx)+'/'+jobname+ '_' + self.resWT + '0' + str(self.resID_ori[1]) + self.resMT+'/'+self.PDBid+'_MT.pdb')
            else:
                os.system('cp '+wild_files[0]+' ../../feature_rank_{}/'.format(idx)+'/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'/'+self.PDBid+'_WT.pdb')
                os.system('cp '+mut_files[0]+' ../../feature_rank_{}/'.format(idx)+'/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'/'+self.PDBid+'_MT.pdb')

            if not os.path.exists('../../feature_jackal_{}/'.format(idx)+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT):
                if len(str(self.resID_ori[1])) == 1:
                    os.mkdir('../../feature_jackal_{}/'.format(idx)+jobname+ '_' + self.resWT + '0' + str(self.resID_ori[1]) + self.resMT)
                else:
                    os.mkdir('../../feature_jackal_{}/'.format(idx)+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT)
            if len(str(self.resID_ori[1])) == 1:
                os.system('cp '+wild_files[0]+' ../../feature_jackal_{}/'.format(idx)+'/'+jobname+ '_' + self.resWT + '0' + str(self.resID_ori[1]) + self.resMT+'/'+self.PDBid+'_WT.pdb')
            else:
                os.system('cp '+wild_files[0]+' ../../feature_jackal_{}/'.format(idx)+'/'+jobname+ '_' + self.resWT + str(self.resID_ori[1]) + self.resMT+'/'+self.PDBid+'_WT.pdb')

    def debugfold_dir(self):
        self.filefold = self.PDBid+'_'+self.muteChain 
        jobname = self.filefold 
        wild_name = '../../alphafold/'+jobname+'_WT'
        mut_name = '../../alphafold/'+jobname+ '_' + self.resWT + str(self.resID[1]) + self.resMT+'_MT'

        if not os.path.exists('../../alphafold/'+jobname+'_WT'):
            print(jobname+ '_WT')

        if not os.path.exists('../../alphafold/'+jobname+ '_' + self.resWT + str(self.resID[1]) + self.resMT):
            print(jobname+ '_' + self.resWT + str(self.resID[1]) + self.resMT)

    def readOtherFASTA(self, ChainID, s):
        fp = open(self.PDBid+'.pdb')
        # this MISS_RES only record the target chain 'ChainID'
        array_RES = {}
        flagMISSRES = False; marker = ''
        for line in fp:
            words = re.split(' |\n', line)
            if len(words) > 4:
                if words[2] == 'MISSING' and words[3] == 'RESIDUES':
                    marker = words[1]
                    flagMISSRES = True
                    break
        if flagMISSRES:
            for _ in range(5): # skip 5 lines
                fp.readline()
            line = re.split(' |\n', fp.readline())
            line = [i for i in line if i]
            while line[1] == marker:
                if line[3] in ChainID:
                    if line[2] in AminoA:
                        array_RES[int(line[4])] = three_to_one(line[2])
                    #elif line[2] != 'HOH':
                    #    self.non_canonical.append(line[2])
                line = re.split(' |\n', fp.readline())
                line = [i for i in line if i]
        fp.close()

        # if array_RES is empty, then jump is not allowed or
        # the largest index of MISSING RESIDUE is less than starter residue
        residue = next(iter(s[0][ChainID])) # what's this? # gives the first residue
        start_idx_PDB  = residue.id[1]
        if len(array_RES) == 0:
            flagMISSRES = False
        else:
            array_RES_idx = list(array_RES.keys()) # find the last index
            last_idx_MISS = array_RES_idx[-1]
            if last_idx_MISS < start_idx_PDB:
                flagMISSRES = False
        # if flagMISSRES = False, then jump index is not allowed

        shift = 0;# last_idx 
        resID_mute_in_ = 0
        for idx, iresidue in enumerate(s[0][ChainID]):
            if iresidue.resname in AminoA:
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue.resname)
                else:
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue.resname)
            elif iresidue.resname in NC_AminoA:
                iresidue_resname = NC_AminoA[iresidue.resname]
                if iresidue.id[2] != ' ': # Here is the insertion code appearing
                    shift += 1
                if flagMISSRES:
                    array_RES[iresidue.id[1]+shift] = three_to_one(iresidue_resname)
                else:
                    array_RES[start_idx_PDB+idx] = three_to_one(iresidue_resname)

        # sort array_RES by index
        array_RES_sorted = {}
        for idx in sorted(array_RES):
            array_RES_sorted[idx] = array_RES[idx]
        #print(array_RES_sorted.keys())

        # get the start and end index, real mutation ID
        array_RES_idx = list(array_RES_sorted.keys())
        idx_s = array_RES_idx[0]; idx_e = array_RES_idx[-1]
        # 0 in array_RES_idx, when flagMISSRES == True
        if 0 not in array_RES_idx and idx_s < 0:
            seqlength = idx_e-idx_s
        else:
            seqlength = idx_e-idx_s + 1

        if seqlength == len(array_RES_idx):
            fasta = ''.join(list(array_RES_sorted.values()))
            return fasta

        # use SEQRES in PDB for fasta file
        array_RES = {}; idx = idx_s
        fp = open(self.PDBid+'.pdb')
        for line in fp:
            if line[:6] == 'SEQRES':
                words = [i for i in re.split(' |\n', line) if i]
                if words[2] == ChainID:
                    for resname in words[4:]:
                        if resname in AminoA:
                            array_RES[idx] = three_to_one(resname)
                            idx += 1
        fp.close()
        fasta = ''.join(list(array_RES.values()))
        return fasta

    def compareSeqFasta(self, ChainID):
        from Bio.SeqIO.FastaIO import SimpleFastaParser
        # if the PDB cannot contains a consecutive FASTA sequence
        self.fasta_PDB = ''

        if not os.path.exists(self.PDBid+'.fasta'):
            os.system('wget https://www.rcsb.org/fasta/entry/'+self.PDBid)
            os.system('mv '+self.PDBid+' '+self.PDBid+'.fasta')

        # values is a tuple (info, fasta)
        # exampel: 1E50_1|Chains A,C,E,G,Q,R[auth I]|CORE-BINDING FACTOR ALPHA SUBUNIT|HOMO SAPIENS
        # want to get ['A', 'C', 'E', 'G', 'Q', 'R'] and 'I'
        # !!! PDBid|Chains A[auth B]| 
        # !!! PDBid|Chains B[auth A]| needs to tell the difference
        # from Bio.SeqIO.FastaIO import SimpleFastaParser to read the fasta file
        ## from BioPython structure, import chains
        #for chain in s[0]:
        #    print(chain.id)
        fasta_file = SimpleFastaParser(open(self.PDBid+'.fasta'))
        fasta = {}
        for values in fasta_file:
            # two ways: 1. only save the fasta that we need; 2. save the fasta for all chains
            #ichain = [i for i in self.Chains if i in fasta_chains][0]
            #self.fasta[ichain] = values[1]
            chain_info = values[0].split('|')[1]
            fasta_chains_temp = re.split(' |\[|\]', chain_info)
            fasta_chains = fasta_chains_temp[1].split(',')
            for i_chain in fasta_chains_temp[2:]:
                fasta_chains += i_chain.split(',')
            for ichain in fasta_chains:
                fasta[ichain] = values[1]
        fasta_target = fasta[ChainID]
        #print(fasta_target)
        #print(self.fasta[ChainID])
        if fasta_target == self.fasta[ChainID]:
            return (True, fasta_target)
        else:
            if len(self.fasta[ChainID]) == len(fasta_target):
                print(self.PDBid, ChainID, 'length same but AA different')
                return (False, fasta_targe)
            elif len(self.fasta[ChainID]) < len(fasta_target):
                for i in range(len(fasta_target)-len(self.fasta[ChainID])+1):
                    if self.fasta[ChainID] == fasta_target[i:i+len(self.fasta[ChainID])]:
                        return (True, fasta_target)
                print(self.PDBid, ChainID, 'self.fasta length less')
                return (False, fasta_target)
            else:
                print(self.PDBid, ChainID, 'self.fasta length longer')
                return (False, fasta_target)

    def readFASTA_(self):
        # Check https://biopython.org/docs/1.75/api/Bio.PDB.Polypeptide.html
        # MISSING RESIDUES are still problem
        from Bio.PDB.Polypeptide import PPBuilder
        ppb = PPBuilder()
        for pp in ppb.build_peptides(self.s, aa_only=False):
            print(pp.get_sequence())
        return # readFASTA_
    
    def transformer(self):
        import torch, esm
        # Load ESM-1b model
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        data = []
        fastaWT, fastaMT = self.fasta['WT'], self.fasta['MT']
        if len(fastaWT) > 1022:
            mutation_pos = self.resID_fasta
            if abs(mutation_pos-1022) < 500:
                fastaWT = fastaWT[-1022:]
                fastaMT = fastaMT[-1022:]
            else:
                fastaWT = fastaWT[:1022]
                fastaMT = fastaMT[:1022]
        data.append(('WT', fastaWT))
        data.append(('MT', fastaMT))
        batch_labels, batch_strs, batch_tokens = batch_converter(data)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results['representations'][33].numpy()

        partner1_WT = []
        partner1_MT = []
        len_seq = len(fastaWT)
        partner1_WT.append(token_representations[-2, 1:len_seq+1].mean(0))
        partner1_WT = np.array(partner1_WT).mean(0)
        len_seq = len(fastaMT)
        partner1_MT.append(token_representations[-1, 1:len_seq+1].mean(0))
        partner1_MT = np.array(partner1_MT).mean(0)

        sequence_representations = []
        sequence_representations.append(partner1_WT)
        sequence_representations.append(partner1_MT)
        #sequence_representations.append(partner1_MT-partner1_WT)
        sequence_representations = np.array(sequence_representations)

        return sequence_representations

def run_struc(PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH):

    #print(PDBid, Chains, muteChain, resID, resWT, resMT, ddG, temp, pH)
    os.system('mkdir ./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')

    if not os.path.exists('./jackal.dir'):
        os.system('cp ../../bin/jackal.dir .')

    s = get_structure(PDBid, Chains, muteChain, resWT, resID, resMT, pH=pH)
    s.generateMutedPDBs()
    s.generateMutedPQRs()
    s.readFASTA()
    #s.writeFoldFASTA()
    #s.generateMutedFoldPDBs()

    for idx in ['001']:
        #if not os.path.exists('../../feature_rank_{}'.format(idx)):
            #os.mkdir('../../feature_rank_{}'.format(idx))
        if not os.path.exists('../../feature_jackal_{}'.format(idx)):
            os.mkdir('../../feature_jackal_{}'.format(idx))

    s.buildfold_dir()
    #print(s.fasta)
    flag_BLAST = True
    flag_MIBPB = True
    if flag_BLAST:
        s.writeFASTA()

    #########################################################################################
    p_WT = Protein(s, 'WT')
    
    p_WT.construct_feature_global()
    p_WT.construct_feature_env()
    if flag_MIBPB:
        #os.system('rm *.arealist')
        #os.system('rm *.areavolume')
        #os.system('rm *.eng')
        #os.system('rm *.englist')
        #os.system('rm wild.pqr')
        #os.system('rm mute.pqr')
        p_WT.construct_feature_MIBPB()
    if flag_BLAST:
        p_WT.runBLAST()
        p_WT.construct_feature_seq()
    
    #----------------------------------------------------------------------------------------
    p_MT = Protein(s, 'MT')
    
    p_MT.construct_feature_global()
    p_MT.construct_feature_env()
    if flag_MIBPB:
        p_MT.construct_feature_MIBPB()
    if flag_BLAST:
        p_MT.runBLAST()
        p_MT.construct_feature_seq()
    feature_aux     = construct_feature_aux(p_WT, p_MT, flag_MIBPB=flag_MIBPB, flag_BLAST=flag_BLAST)
    feature_aux_inv = construct_feature_aux(p_MT, p_WT, flag_MIBPB=flag_MIBPB, flag_BLAST=flag_BLAST)
    #----------------------------------------------------------------------------------------
    print('auxiliary feature size: ', feature_aux.shape)
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT
    OutFile = open(filename+'_aux.npy', 'wb')
    np.save(OutFile, feature_aux)
    OutFile.close()

    filename_inv = PDBid+'_'+Chains+'_'+resMT+'_'+resID+'_'+resWT
    OutFile = open(filename_inv+'_aux.npy', 'wb')
    np.save(OutFile, feature_aux_inv)
    OutFile.close()
    
    p_WT.rips_complex()
    p_WT.alpha_complex()
    p_MT.rips_complex()
    p_MT.alpha_complex()
    PH_feature = construct_feature_PH(p_WT, p_MT)
    PH12_feature = construct_feature_PH12(p_WT, p_MT)
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT
    Top_feature = np.concatenate((PH_feature, PH12_feature), axis=0)
    print('Topological feature size: ', Top_feature.shape)
    OutFile = open(filename+'_Top.npy', 'wb')
    np.save(OutFile, Top_feature)
    OutFile.close()

    PH_feature = construct_feature_PH(p_MT, p_WT)
    PH12_feature = construct_feature_PH12(p_MT, p_WT)
    filename_inv = PDBid+'_'+Chains+'_'+resMT+'_'+resID+'_'+resWT
    Top_feature = np.concatenate((PH_feature, PH12_feature), axis=0)
    print('Topological feature size: ', Top_feature.shape)
    OutFile = open(filename_inv+'_Top.npy', 'wb')
    np.save(OutFile, Top_feature)
    OutFile.close()

    os.chdir('../')

def run_transformer(PDBid, Chains, muteChain, resID, resWT, resMT, ddG, temp, pH):

    #print(PDBid, Chains, muteChain, resID, resWT, resMT, ddG, temp, pH)
    os.system('mkdir ./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')

    if not os.path.exists('./jackal.dir'):
        os.system('cp ../../bin/jackal.dir .')

    s = get_structure(PDBid, Chains, muteChain, resWT, resID, resMT, pH=pH)
    s.generateMutedPDBs()
    s.generateMutedPQRs()
    s.readFASTA()
    #print(s.fasta)
    tf_feature = s.transformer()
    print('Transformer feature size: ', tf_feature.shape)
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT
    OutFile = open(filename+'_Transformer.npy', 'wb')
    np.save(OutFile, tf_feature)
    OutFile.close()

    tf_feature_WT, tf_feature_MT = tf_feature[0], tf_feature[1]
    tf_feature_inv = []
    tf_feature_inv.append(tf_feature_MT)
    tf_feature_inv.append(tf_feature_WT)
    tf_feature_inv = np.array(tf_feature_inv)
    print('Transformer feature size: ', tf_feature_inv.shape)
    filename_inv = PDBid+'_'+Chains+'_'+resMT+'_'+resID+'_'+resWT
    OutFile = open(filename_inv+'_Transformer.npy', 'wb')
    np.save(OutFile, tf_feature_inv)
    OutFile.close()

    os.chdir('../')

def run_Lap(PDBid, Chains, muteChain, resID, resWT, resMT, ddG, temp, pH):

    #print(PDBid, Chains, muteChain, resID, resWT, resMT, ddG, temp, pH)
    os.system('mkdir ./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')
    os.chdir('./'+PDBid+'_'+Chains+'_'+resWT+resID+resMT+'/')

    if not os.path.exists('./jackal.dir'):
        os.system('cp ../../bin/jackal.dir .')

    s = get_structure(PDBid, Chains, muteChain, resWT, resID, resMT, pH=pH)
    s.generateMutedPDBs()
    s.generateMutedPQRs()
    s.readFASTA()
    p_WT = Protein(s, 'WT')
    p_MT = Protein(s, 'MT')
    WT_feat = p_WT.rips_complex_spectra()
    MT_feat = p_MT.rips_complex_spectra()
    tf_feature = np.concatenate((WT_feat, MT_feat))
    print('Laplacian feature size: ', tf_feature.shape)
    filename = PDBid+'_'+Chains+'_'+resWT+'_'+resID+'_'+resMT
    OutFile = open(filename+'_Lap.npy', 'wb')
    np.save(OutFile, tf_feature)
    OutFile.close()

    tf_feature_inv = np.concatenate((MT_feat, WT_feat))
    print('Laplacian feature size: ', tf_feature_inv.shape)
    filename_inv = PDBid+'_'+Chains+'_'+resMT+'_'+resID+'_'+resWT
    OutFile = open(filename_inv+'_Lap.npy', 'wb')
    np.save(OutFile, tf_feature_inv)
    OutFile.close()

    os.chdir('../')

def Gen_MIBPB(PDBid, Chain, muteChain, resID, resWT, resMT, ddG, temp, pH):

    #os.system('ls')
    os.chdir(PDBid+'_'+Chain+'_'+resWT+resID+resMT+'/')

    os.system('rm *.arealist')
    os.system('rm *.areavolume')
    os.system('rm *.eng')
    os.system('rm *.englist')
    os.system('rm wild.pqr')
    os.system('rm mute.pqr')

    pdbfile = PDBid+'_WT.pdb'
    pqrfile = PDBid+'_WT.pqr'
    
    Name = PDBid+'_'+str(pH)
    if not os.path.exists(Name+'.englist') or not os.path.exists(Name+'.eng') \
    or not os.path.exists(Name+'.arealist') or not os.path.exists(Name+'.areavolume'):
        filepqr = open(pqrfile,'r')
        newpqr = open('wild.pqr', 'w')
        for line in filepqr:
            newline = line
            if line[0:4] == 'ATOM':
                if line[26:27] != " ":
                    newline = line[0:26] + "  " + line[28:]
            newpqr.write(newline)
        filepqr.close()
        newpqr.close()
        os.system('mibpb5 wild h=0.5')
        os.system('mv partition_area.txt '+Name+'.arealist')
        os.system('mv area_volume.dat '+Name+'.areavolume')
        os.system('mv wild.eng '+Name+'.eng')
        os.system('mv wild.englist '+Name+'.englist')
    print('Calculating wild type features')

    pdbfile = PDBid+'_MT.pdb'
    pqrfile = PDBid+'_MT.pqr'

    Name = PDBid+'_'+str(pH)+'_mut'
    if not os.path.exists(Name+'.englist') or not os.path.exists(Name+'.eng') or \
    not os.path.exists(Name+'.arealist') or not os.path.exists(Name+'.areavolume'):
        filepqr = open(pqrfile, 'r')
        newpqr = open('mute.pqr', 'w')
        for line in filepqr:
            newline = line
            if line[0:4] == 'ATOM':
                if line[26:27] != " ":
                    newline = line[0:26] + "  " + line[28:]
            newpqr.write(newline)
        filepqr.close()
        newpqr.close()
        os.system('mibpb5 mute h=0.5')
        os.system('mv partition_area.txt '+Name+'.arealist')
        os.system('mv area_volume.dat '+Name+'.areavolume')
        os.system('mv mute.eng '+Name+'.eng')
        os.system('mv mute.englist '+Name+'.englist')
    print('Calculating mutant features')

    os.chdir('../')

time1 = time.time()
if not os.path.exists("data"):
    os.system("mkdir data")

df = pd.read_csv("./mutsol/mutsol.csv")
data = df.to_numpy()

mutsol = []
for i in range(len(data)):
    tmp = data[i]
    PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH = str(tmp[3]), "A", "A", tmp[1][1:-1], tmp[1][0], tmp[1][-1], int(tmp[2]), 25, 7
    mutsol.append([PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH])

    #print(PDBid, Chain, resID, resWT, resMT, ddG, temp, pH)

np.save("./mutsol/mutsol.npy", mutsol)

data = np.load("./mutsol/mutsol.npy", allow_pickle=True)

ll = []
os.system('mkdir '+ './mutsol/feature/')
os.chdir('./mutsol/feature/')

start, end = int(sys.argv[1]), int(sys.argv[2])
#idx, code = sys.argv[3], sys.argv[4]
#idx = sys.argv[3]
inp = []

for i in range(start, end):
    tmp = data[i]
    #print(tmp)
    PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH = str(tmp[0]), tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], int(tmp[6]), tmp[7], tmp[8]
    
    inp.append((PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH))
    #print(PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH)
    Gen_MIBPB(PDBid, Chain, muteChain, resID, resWT, resMT, sol, temp, pH)
    run_struc(PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH)
    run_transformer(PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH)
    run_Lap(PDBid, Chains, muteChain, resID, resWT, resMT, sol, temp, pH)