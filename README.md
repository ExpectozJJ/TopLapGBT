# Integration of persistent Laplacian and pre-trained transformer for protein solubility changes upon mutation
This code is for Integration of persistent Laplacian and pre-trained transformer for protein solubility changes upon mutation. 
******

## Python Environment needed
- fair-esm                  2.0.0
- numpy                     1.23.5
- scipy                     1.11.3
- scikit-learn              1.3.2
- python                    3.10.12
- Biopython                 1.80
- Softwares to be installed for ./bin folder (See README at bin)

## File description
There are five folders after extracted. 
* **bin** 
    * jackal.dir
    * jackal_64bit
    * PPIprepare
    * PPIstructure
    * profix
    * scap
    * **SPIDER2_local**
    * mibpb5 
    * MS_Intersection 
* **dataset**
    * alphafold_pdb.zip (at root directory)
* **mutsol**
    * mutsol.csv
    * mutsol.npy     
* **model and feature generation**
    * mutsol_build_features.py
    * mutsol_build_cv.py
    * mutsol_blindtest.py
    * mutsol_10cv.py

## Feature generation
The dependencies libraries (the current version is only test on linux system):

* BLAST+/2.10.1
* dssp/3.1.4
* vmd (should be a global call)
* pdb2pqr (should be a global call)
* mibpb5 (should be a globale call, which can be used from https://weilab.math.msu.edu/MIBPB/)
