import numpy as np
from scipy.linalg import hankel
import math

def unfold(tensor,n):
    size = np.array(tensor.shape)
    N = size.shape[0]
    I = int(size[n])
    J = int(int(np.prod(size)) / int(I))
    pmt=np.array(range(n,n+1))
    pmt=np.append(pmt,range(0,n))
    pmt=np.append(pmt,range(n+1,N)).astype(np.int)
    return np.reshape(np.transpose(tensor, pmt),[I, J])

def fold(matrix,n,size_t_ori):
    N = np.array(size_t_ori).shape[0]
    size_t_pmt = np.concatenate([size_t_ori[n:(n+1)],size_t_ori[0:n],size_t_ori[(n+1):N]], axis=0)
    pmt = np.array(range(1,n+1))
    pmt = np.append(pmt,range(0,1))
    pmt = np.append(pmt,range(n+1,N)).astype(np.int)
    return np.transpose(np.reshape(matrix,size_t_pmt),pmt)

def make_duplication_matrix(T,tau):
    H = hankel(range(tau),range(tau-1,T))
    T2= np.prod(H.shape)
    h = np.reshape(H,[1,T2])
    h2= np.array([range(T2)])
    index = np.concatenate([h,h2],axis=0)
    S = np.zeros([T,T2], dtype='uint64')
    S[tuple(index)]=1
    return S.T

def tmult(tensor,matrix,n):
    size = np.array(tensor.shape)
    size[n] = matrix.shape[0]
    return fold(np.matmul(matrix,unfold(tensor,n)),n,size)

def hankel_tensor(x,TAU):
    N = len(TAU)
    N2= N*2
    T2  = np.zeros([N,2],dtype='uint64')
    S = list()
    Hx = x
    for n in range(N):
        tau   = TAU[n]
        T     = x.shape[n]
        T2[n,:] = [tau,T-tau+1]
        S.append(make_duplication_matrix(x.shape[n],TAU[n]))
        Hx = tmult(Hx,S[n],n)
    size_h_tensor = np.reshape(T2,[N2,])
    Hx = np.reshape(Hx,size_h_tensor)
    return Hx, S

def hankel_tensor_adjoint(Hx,S):
    N = len(S)
    size_h_tensor = np.zeros([N,], dtype='uint64')
    for n in range(N):
        size_h_tensor[n] = S[n].shape[0]
    Hx = np.reshape(Hx,size_h_tensor)
    for n in range(N):
        Hx = tmult(Hx,S[n].T,n)
    return Hx
    
