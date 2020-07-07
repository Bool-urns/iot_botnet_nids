"""
Created on Tue Feb  6 12:33:42 2018

@authors: Radu and Ioana Dogaru 
Revised: May 8, 2018 ; March 7, 2019 ; April 15 2019; May 18 2019; 

Implements the algorithm described in 
[1] Radu Dogaru and Ioana Dogaru, "Optimization of super fast support 
vector classifiers using Python and acceleration of RBF computations",
COMM-2018.  


Please cite the above reference when using it 

=============================================
Resuted trained model: inW, Wad, typ, raza 
=============================================

inW is a matrix with m RBF center vectors, each center (support vector)
being selected from the N training samples 

Wad is the output matrix (linear output layer) with coefficients 0 or 1 
(thus very convenient to implement in HW)

raza = radius of the RBF function implementing the hidden neurons 

typ = the nature of the RBF function (typ=1 is piecewise-linear and HW friendly)


SFSVC is well suited for embbeded system applications where RBF 
neurons are implemented in the piecewise linear fashion (no exponential) i.e. mode=1
The system is a "weightless" ones since the output (Adaline layer) has only 0 / 1 values 
Aslo convenient for HW implements. 

In general, may replace SVM with certain advantages regarding learning speed 
(particularly for large databases) and the possibility for implementing
resource-constrained classifiers. 

The computation of the RBF layer is carefully optimized for platfomes 
supporting Intel's MKL (so prediction with SFSVC is two orders of magnitude 
faster than SCIKIT SVM for the same number of support vectors) 

Try the same data set using the svm.py file (included) to compare training
and prediction speed !!!


Usage: fill in the proper algorithm parameters and start tuning the radius 
until best performance is obtained ; The latest model is always saved in .mat 
file last_model.mat  


"""

import numpy as np 
import time 
import scipy.io as sio

#===============  ALGORITHM  PARAMETERS =====================================

#nume='optd64'  #  Dataset files _train.mat _test.mat 
nume='usps'
# several such pairs are provided here  
# http://atm.neuro.pub.ro/radu_d/html/09_10/ici2009/k-lab/date-ici.html

'''
raza=25.4; raza=float(raza) # Radius (start with bigger values and decreas until optimal performance)
prag=1; prag=float(prag)  # Activation (overlap) threshold - start with 1  
typ=1  # RBF function: 2 - Gaussian , # 1- triangular (best suited for HW)
alea=0 # if 1 - input samples are radnomized -> will slightly influence performance (like in ELM)
first_samples = 0 # 0  all samples are considered  ; n>0  - first n samples are considered (useful for faster radius tuning)
'''

# ======================================================================

# Optimized (Intel MKL based) computation of RBF layers 
# Best speed obtained for float32 variables 
def rbf_layer_mkl(Samples,inW,raza,typ):
    N=np.size(Samples,0)
    n=np.size(Samples,1)
    m=np.size(inW,1)
    Ocol = np.ones((n,1), dtype = np.float32)
    Qlin = Ocol.T
    # (a-b)^2 equival with a^2+b^2-2a*b (main computation a*b)
    d=np.repeat(np.dot(Samples*Samples,Ocol),m,axis=1)+np.repeat(np.dot(Qlin,inW*inW),N,axis=0)-2*np.dot(Samples,inW)
    if typ==1:
        d=1-d/(raza*2.5066)
        Hrbf=(d+np.abs(d))/2.0
    elif typ==2:
        kgauss=-1/(2*raza*raza)
        Hrbf=np.exp(kgauss*d*d) #  
        
    return Hrbf
    
def novelty_layer_compute(Sa,r,pra,ty):
# With given parameters (radius (r), threshold (pra) and RBF type (ty))
# Computes the "tix" (indexes) of the "support vectors" selected from 
# the Samples batch (Sa) 
# Returns the selected support vectors 
    
    N=np.size(Sa,0)
    tix=np.array([]).astype(np.int32)
    # First support vector is associated with first index 
    tix=np.append(tix,0)
    # Support vector selection loop 
    for i in range(1,N):
        # Compute the RBF layer activity for the actual inW
        # when the current sample (only one) is applied 
        # - using rbf_layer_mkl 
        Hid=rbf_layer_mkl(Sa[i:(i+1),:],Sa[tix,:].T, r, ty)
        Activity=np.sum(Hid)
        if Activity<pra:
            # A new Support vector is added as the current sample
            tix=np.append(tix,i)
    return Sa[tix,:].T
             
def sfsvc_train(Sa,La,ra,pra,ty,alea):
# Implements the "supervized" (class based) training a.k.a SFSVC 
# For each class a selection of support vectors is found using novelty_layer 
# and the number of neurons for each class is computed 
# Returns the input layer matrix (inW) and the list of neurons for each class 
    
    M=np.max(Labels)
    N=np.size(Samples,0)  
    
    if alea==1:
        ixpe=np.random.permutation(N)
        Sa=Sa[ixpe,:]
        La=La[:,ixpe]        
        
    k=0 # first class - find the support vectors 
    ixk=np.where(La==(k+1))
    Sk=Sa[ixk[1],:] 
    inW=novelty_layer_compute(Sk,ra,pra,ty)
    nk=np.size(inW,1)  # number of neurons in class k 
    neuroni=np.array([]).astype(np.int32)   
    neuroni=np.append(neuroni,nk)
    for k in range(1,M):      # next 2-M classes - find support vectors
        ixk=np.where(La==(k+1)) # select class k indices
        Sk=Sa[ixk[1],:]  # Sk is Samples of class k 
        rbfk=novelty_layer_compute(Sk,ra,pra,ty) 
        nk=np.size(rbfk,1)  # number of neurons in class k 
        neuroni=np.append(neuroni,nk)
        inW=np.append(inW,rbfk,axis=1)
    return (inW, neuroni)
            
def create_outw(inW,neur):
# Creates the output layer (outW) composed of bunary weights 0 or 1 
# Returns outW 
    
    M=np.size(neur)
    m=np.size(inW,1)
    outW=np.zeros((m,M)).astype('float32')
    l1=0; l2=0; nk=0
    for k in range(M):
        l1=l1+nk; 
        nk=neur[k]; 
        l2=l2+nk; 
        outW[range(l1,l2),k]=1    
    return outW
    

#-------------------  MAIN  ---------------------------
# Resulting model is given by:  tip , radius, inW, outW 
#--------------------------------------------------------
#  reads the training set
def sfsvc_classify(train_labels, train_samples, test_labels, test_samples):
    raza=25.4; raza=float(raza) # Radius (start with bigger values and decreas until optimal performance)
    prag=1; prag=float(prag)  # Activation (overlap) threshold - start with 1  
    typ=1  # RBF function: 2 - Gaussian , # 1- triangular (best suited for HW)
    alea=0 # if 1 - input samples are radnomized -> will slightly influence performance (like in ELM)
    first_samples = 0 # 0  all samples are considered  ; n>0  - first n samples are considered (useful for faster radius tuning)
    
    timer = time.time()
    #db=sio.loadmat(nume+'_train.mat')
    #Samples=db['Samples'].astype('float32')
    Samples = train_samples.astype('float32')
    Samples=Samples.T

    #Labels=db['Labels'].astype('int8')
    Labels=train_labels.astype('int8')
    N=np.size(Samples,0)
    n=np.size(Samples,1)
    M=np.max(Labels)
    runtime = time.time() - timer
    print( " load train data time: %f s" % runtime)

    if first_samples>0:
        if first_samples>N: 
            first_samples=N
        N=first_samples
        Samples=Samples[0:N,:]
        Labels=Labels[:,0:N]


    # ================ TRAIN SFSV =======================
    # Implements the Super Fast Support Vector Classifiers train 
    # In fact a selection of support vectors form training set 

    timer = time.time()
    (inW,neur)=sfsvc_train(Samples,Labels,raza,prag,typ,alea)
    Wad=create_outw(inW,neur)
    runtime = time.time() - timer
    print( " TRAINING time: %f s" % runtime)

    #=========LOAD Test Set ================================

    timer = time.time()
    #db=sio.loadmat(nume+'_test.mat')
    #Samples=db['Samples'].astype('float32')
    Samples = test_samples.astype('float32')
    Samples=Samples.T
    #Labels=db['Labels'].astype('int8')
    Labels=test_labels.astype('int8')
    N=np.size(Samples,0)
    n=np.size(Samples,1)
    M=np.max(Labels)

    runtime = time.time() - timer
    print( " load test data time: %f s" % runtime)

    #===================TEST phase (retrieval) ==========================
    timer = time.time()
    Hrbfk=rbf_layer_mkl(Samples,inW,1*raza,typ)
    # Here one can try altered RBF layers (e.g. with typ=1)
    # to improve implementations 
    # Also radius may change from what was used in training phase 
    # 1.2 coefficnet may help .. 

    Scores=np.dot(Hrbfk,Wad)
    runtime = time.time() - timer
    print( " PREDICTION (test data) time: %f s" % runtime)

    #================= Evaluate Accuracy for the Test set  =============================

    Conf=np.zeros((M,M),dtype='int16')
    for i in range(N):
        # gasire pozitie clasa prezisa 
        ix=np.where(Scores[i,:]==np.max(Scores[i,:]))
        ixx=np.array(ix)
        pred=int(ixx[0,0])
        actual=Labels[0,i]-1
        Conf[actual,pred]+=1
    accuracy=100.0*np.sum(np.diag(Conf))/np.sum(np.sum(Conf))
    print("Confusion matrix: ")
    print(Conf)
    print("Accuracy: %f" %accuracy)
    print( "Neurons for each class layer: ", neur)
    nr_neuroni=np.sum(neur)
    print( "Total number of hidden layer: %d" %nr_neuroni)
    print( "RBF type:  (1=Triangular; 2=Gaussian): %d" %typ)
    print( "radius: %f" %raza)
    print( "overlapping factor: %f" %prag)
    print( "alea: %d" %alea)
    # ========================================  SAVE THE MODEL ========================
    # Salvare LAST_MODEL 
    last_model={'inW':inW, 'Wad':Wad, 'tip':typ, 'raza':raza, 'accuracy':accuracy, 'nume':nume}
    #sio.savemat('last_model.mat',last_model)
