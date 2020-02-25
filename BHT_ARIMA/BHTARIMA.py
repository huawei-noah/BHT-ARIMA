
import copy

import tensorly as tl
import scipy as sp
import numpy as np
from tensorly.decomposition import tucker

from .util.MDT import MDTWrapper
from .util.functions import fit_ar_ma, svd_init


class BHTARIMA(object):
    """BHT-ARIMA, a tensor-base ARIMA algorithm

    This algorithm can forecast multiple series at the same time based on  capturing the 
    intrinsic correlations

    Paramters
    ---------
        ts : np.ndarray, shape (I1, I2, ..., IT)
            Training data, a tensor time series(tensor-mode) with shape of I1*I2*...*IN*T
        
        p : int
            The order of AR algorithm
        
        d : int 
            The order of difference

        q : int
            The order of MA algorithm

        taus : list[int]
            Ranks of MDT tensorization, recommand the first element is the num of series
        
        Rs : list[int]
            Ranks of Tucker decomposition
        
        K : int
            Iterations of training
        
        tol : float
            Convergence threshold, it should be samller than 1.0
        
        seed : int, default None
            The random number seed
        
        Us_mode : int, default 4
            The mode of orthogonality
                - 2 : releaxed-orthogonality
                - 4 : full-orthogonality
        
        verbose : int, default 0
            The level of displaying intermediate info
                - 0 : not display
                - 1 : display info of each iteration
        
        convergence_loss : boolean, default False
            Whether return the convergence loss of each iteration
    
    """
    
    def __init__(self,ts, p, d, q, taus, Rs, K, tol, seed=None, Us_mode=4, \
        verbose=0, convergence_loss=False):
        """store all parameters in the class and do checking on taus"""
        
        self._ts = ts
        self._ts_ori_shape = ts.shape
        self._N = len(ts.shape) - 1
        self.T = ts.shape[-1]
        self._p = p
        self._d = d
        self._q = q
        self._taus = taus
        self._Rs = Rs
        self._K = K
        self._tol = tol
        self._Us_mode = Us_mode
        self._verbose = verbose
        self._convergence_loss = convergence_loss
        
        if seed is not None:
            np.random.seed()
        
        # check Rs parameters
        M = 0
        for dms,tau in zip(self._ts_ori_shape, taus):
            if dms == tau:
                M += 1
            elif dms > tau:
                M += 2
        if M-1 != len(Rs):
            raise ValueError("the first element of taus should be equal to the num of series")
    
    def _forward_MDT(self, data, taus):
        self.mdt = MDTWrapper(data,taus)
        trans_data = self.mdt.transform()
        self._T_hat = self.mdt.shape()[-1]
        return trans_data, self.mdt
    
    def _initilizer(self, T_hat, Js, Rs, Xs):
        
        # initilize Us
        U = [ np.random.random([j,r]) for j,r in zip( list(Js), Rs )]

        # initilize es
        begin_idx = self._p + self._q
        es = [ [ np.random.random(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]

        return U, es

    def _test_initilizer(self, trans_data, Rs):
        
        T_hat = trans_data.shape[-1]
        # initilize Us
        U = [ np.random.random([j,r]) for j,r in zip( list(trans_data.shape)[:-1], Rs )]

        # initilize es
        begin_idx = self._p + self._q
        es = [ [ np.zeros(Rs) for _ in range(self._q)] for t in range(begin_idx, T_hat)]
        return U, es
    
    def _initilize_U(self, T_hat, Xs, Rs):

        haveNan = True
        while haveNan:
            factors = svd_init(Xs[0], range(len(Xs[0].shape)), ranks=Rs)
            haveNan = np.any(np.isnan(factors))
        return factors  
                     
    def _inverse_MDT(self, mdt, data, taus, shape):
        return mdt.inverse(data, taus, shape)
    
    def _get_cores(self, Xs, Us):
        cores = [ tl.tenalg.multi_mode_dot( x, [u.T for u in Us], modes=[i for i in range(len(Us))] ) for x in Xs]
        return cores
    
    def _estimate_ar_ma(self, cores, p, q):
        cores = copy.deepcopy(cores)
        alpha, beta = fit_ar_ma(cores, p, q)
        
        return alpha, beta
    
    def _get_fold_tensor(self, tensor, mode, shape):
        if isinstance(tensor,list):
            return [ tl.base.fold(ten, mode, shape) for ten in tensor ]
        elif isinstance(tensor, np.ndarray):
            return tl.base.fold(tensor, mode, shape)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")

    def _get_unfold_tensor(self, tensor, mode):
        
        if isinstance(tensor, list):
            return [ tl.base.unfold(ten, mode) for ten in tensor]
        elif isinstance(tensor, np.ndarray):
            return tl.base.unfold(tensor, mode)
        else:
            raise TypeError(" 'tensor' need to be a list or numpy.ndarray")   

    def _update_Us(self, Us, Xs, unfold_cores, n):

        T_hat = len(Xs)
        M = len(Us)
        begin_idx = self._p + self._q

        H = self._get_H(Us, n)
        # orth in J3
        if self._Us_mode == 1:
            if n<M-1:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
            else:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
        # orth in J1 J2
        elif self._Us_mode == 2:
            if n<M-1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        # no orth      
        elif self._Us_mode == 3:
            As = []
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            a = sp.linalg.pinv(np.sum(As, axis=0))
            b = np.sum(Bs, axis=0)
            temp = np.dot(a, b)
            Us[n] = temp / np.linalg.norm(temp)
        # all orth
        elif self._Us_mode == 4:
            Bs = []
            for t in range(begin_idx, T_hat):
                unfold_X = self._get_unfold_tensor(Xs[t], n)
                Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
            b = np.sum(Bs, axis=0)
            #b = b.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
            U_, _, V_ = np.linalg.svd(b, full_matrices=False)
            Us[n] = np.dot(U_, V_)
        # only orth in J1
        elif self._Us_mode == 5:
            if n==0:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        # only orth in J2
        elif self._Us_mode == 6:
            if n==1:
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                b = np.sum(Bs, axis=0)
                U_, _, V_ = np.linalg.svd(b, full_matrices=False)
                Us[n] = np.dot(U_, V_)
            else:
                As = []
                Bs = []
                for t in range(begin_idx, T_hat):
                    unfold_X = self._get_unfold_tensor(Xs[t], n)
                    As.append(np.dot(np.dot(unfold_X, H.T), np.dot(unfold_X, H.T).T))
                    Bs.append(np.dot(np.dot(unfold_X, H.T), unfold_cores[t].T))
                a = sp.linalg.pinv(np.sum(As, axis=0))
                b = np.sum(Bs, axis=0)
                temp = np.dot(a, b)
                Us[n] = temp / np.linalg.norm(temp)
        return Us
    
    def _update_Es(self, es, alpha, beta, unfold_cores, i, n):
        
        T_hat = len(unfold_cores)
        begin_idx = self._p + self._q

        As = []
        for t in range(begin_idx, T_hat):
            a = np.sum([alpha[ii] * unfold_cores[:t][-(ii+1)] for ii in range(self._p)] , axis=0)
            b = np.sum([beta[j] * self._get_unfold_tensor(es[t-begin_idx][-(j+1)], n)  for j in range(self._q) if i!=j ] , axis=0)
            As.append(unfold_cores[t] - a + b)
        E = np.sum(As, axis=0)
        for t in range(len(es)):
            es[t][i] = self._get_fold_tensor(E / (2*(begin_idx - T_hat) * beta[i]), n, es[t][i].shape)
        return es

    def _compute_convergence(self, new_U, old_U):
        
        new_old = [ n-o for n, o in zip(new_U, old_U)]
        
        a = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_old], axis=0)
        b = np.sum([np.sqrt(tl.tenalg.inner(e,e)) for e in new_U], axis=0)
        return a/b
    
    def _tensor_difference(self, d, tensors, axis):
        """
        get d-order difference series
        
        Arg:
            d: int, order
            tensors: list of ndarray, tensor to be difference
        
        Return:
            begin_tensors: list, the first d elements, used for recovering original tensors
            d_tensors: ndarray, tensors after difference
        
        """
        d_tensors = tensors
        begin_tensors = []

        for _ in range(d):
            begin_tensors.append(d_tensors[0])
            d_tensors = list(np.diff(d_tensors, axis=axis))
        
        return begin_tensors, d_tensors
    
    def _tensor_reverse_diff(self, d, begin, tensors, axis):
        """
        recover original tensors from d-order difference tensors
        
        Arg:
            d: int, order
            begin: list, the first d elements
            tensors: list of ndarray, tensors after difference
        
        Return:
            re_tensors: ndarray, original tensors
        
        """
        
        re_tensors = tensors      
        for i in range(1, d+1):
            re_tensors = list(np.cumsum(np.insert(re_tensors, 0, begin[-i], axis=axis), axis=axis))
         
        return re_tensors
    
    def _update_cores(self, n, Us, Xs, es, cores, alpha, beta, lam=1):

        begin_idx = self._p + self._q
        T_hat = len(Xs)
        unfold_cores = self._get_unfold_tensor(cores, n)
        H = self._get_H(Us, n)
        for t in range(begin_idx, T_hat):
            unfold_Xs = self._get_unfold_tensor(Xs[t], n)
            b = np.sum([ beta[i] * self._get_unfold_tensor(es[t-begin_idx][-(i+1)], n) for i in range(self._q)], axis=0)
            a = np.sum( [ alpha[i] * self._get_unfold_tensor(cores[t-(i+1)], n) for i in range(self._p)], axis=0 )
            unfold_cores[t] = 1/(1+lam) * (lam * np.dot( np.dot(Us[n].T, unfold_Xs), H.T) + a - b)
        return unfold_cores

    def _get_Xs(self, trans_data):

        T_hat = trans_data.shape[-1]
        Xs = [ trans_data[..., t] for t in range(T_hat)]

        return Xs

    def _get_H(self, Us, n):

        Hs = tl.tenalg.kronecker([u.T for u, i in zip(Us[::-1], reversed(range(len(Us)))) if i!= n ])
        return Hs
    
    def run(self):
        """run the program

        Returns
        -------
        result : np.ndarray, shape (num of items, num of time step +1)
            prediction result, included the original series

        loss : list[float] if self.convergence_loss == True else None
            Convergence loss

        """

        
        result, loss = self._run()
        
        if self._convergence_loss:
            
            return result, loss            
        
        return result, None

    def _run(self):
        
        # step 1a: MDT
        # transfer original tensor into MDT tensors
        trans_data, mdt = self._forward_MDT(self._ts, self._taus)
        
        Xs = self._get_Xs(trans_data)

        if self._d!=0:
            begin, Xs = self._tensor_difference(self._d, Xs, 0)

        # for plotting the convergence loss figure
        con_loss = []
        
        # Step 2: Hankel Tensor ARMA based on Tucker-decomposition

        # initialize Us
        Us, es = self._initilizer(len(Xs), Xs[0].shape, self._Rs, Xs)

        for k in range(self._K):

            old_Us = Us.copy()
            
            # get cores
            cores = self._get_cores(Xs, Us)
            #print(cores)
            # estimate the coefficients of AR and MA model
            alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)
            for n in range(len(self._Rs)):
                
                cores_shape = cores[0].shape
                unfold_cores = self._update_cores(n, Us, Xs, es, cores, alpha, beta, lam=1)
                cores = self._get_fold_tensor(unfold_cores, n, cores_shape)
                # update Us 
                Us = self._update_Us(Us, Xs, unfold_cores, n)
                
                for i in range(self._q):

                    # update Es
                    es = self._update_Es(es, alpha, beta, unfold_cores, i, n)

            # convergence check:
            convergence = self._compute_convergence(Us, old_Us)
            con_loss.append(convergence)
            
            if k%10 == 0:
                if self._verbose == 1:             
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    #print("alpha: {}, beta: {}".format(alpha, beta))

            if self._tol > convergence:
                if self._verbose == 1: 
                    print("iter: {}, convergence: {}, tol: {:.10f}".format(k, convergence, self._tol))
                    print("alpha: {}, beta: {}".format(alpha, beta))
                break

   
        # Step 3: Forecasting
        #get cores
        cores = self._get_cores(Xs, Us)           
        alpha, beta = self._estimate_ar_ma(cores, self._p, self._q)
        
        new_core = np.sum([al * core for al, core in zip(alpha, cores[-self._p:][::-1])], axis=0) - \
                    np.sum([be * e for be, e in zip(beta, es[-1][::-1])], axis=0)

        new_X = tl.tenalg.multi_mode_dot(new_core, Us)
        Xs.append(new_X)

        if self._d != 0:
            Xs = self._tensor_reverse_diff(self._d, begin, Xs, 0)
        mdt_result = Xs[-1]

        # Step 4: Inverse MDT
        # get orignial shape
        fore_shape = list(self._ts_ori_shape)
        merged = []
        for i in range(trans_data.shape[-1]):
            merged.append(trans_data[...,i].T)
        merged.append(mdt_result.T)
        
        merged = np.array(merged)

        mdt_result = merged.T
        
        # 1-step extension (time dimension)
        fore_shape[-1] += 1
        fore_shape = np.array(fore_shape)

        # inverse MDT 
        result = self._inverse_MDT(mdt, mdt_result, self._taus, fore_shape)
        
        
        return result, con_loss