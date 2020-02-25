import numpy as np
import BHT_ARIMA.util.MDT_functions as mdt

class MDTWrapper(object):
    
    def __init__(self,data,tau=None):
        self._data = data.astype(np.float32)
        self._ori_data = data.astype(np.float32)
        self.set_tau(tau)
        is_transformed = False
        self._ori_shape = data.shape
        pass
    
    def set_tau(self, tau):
        if isinstance(tau, np.ndarray):
            self._tau = tau
        elif isinstance(tau, list):
            self._tau = np.array(tau)
        else:
            raise TypeError(" 'tau' need to be a list or numpy.ndarray")
        
    
    def get_tau(self):
        return self._tau
    
    def shape(self):
        return self._data.shape
    
    def get_data(self):
        return self._data
    
    def get_ori_data(self):
        return self._ori_data
    
    def transform(self, tau=None):       
        _tau = tau if tau is not None else self._tau
        result, S = mdt.hankel_tensor(self._data, _tau)
        self.is_transformed = True
        #print("before squeeze: ", result.shape)
        axis_dim = tuple(i for i, dim in enumerate(result.shape) if dim==1 and i!=0)
        result = np.squeeze(result,axis=axis_dim)
        #print("after squeeze: ", result.shape)
        self._data = result
        return result
    
    def inverse(self, data=None, tau=None, ori_shape=None):
        _tau = tau if tau is not None else self._tau
        _ori_shape = ori_shape if ori_shape is not None else self._ori_shape
        _data = data if data is not None else self._data
        O = np.ones(_ori_shape, dtype='uint8')
        Ho, S = mdt.hankel_tensor(O.astype(np.float32), _tau)
        D = mdt.hankel_tensor_adjoint(Ho, S)
        
        result = np.divide(mdt.hankel_tensor_adjoint(_data, S), D)
        self.is_transformed = False
        self._data = result
        return result
    
    def predict(self):
        '''
        # To do:
        # predict function
        '''
        pass

if __name__ == "__main__":

    X = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20.])

    data = np.reshape(X, [4, 5])
    tau = np.array([1,3])

    m = MDTWrapper(data, tau)
    print("original data: \n", m.get_data())
    print("tau: \n", m.get_tau())
    print("transformed data: \n", m.transform())
    print("data transformed? \n", m.is_transformed)
    print("inverse data: \n", m.inverse())

