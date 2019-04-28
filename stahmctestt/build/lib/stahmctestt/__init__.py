import numpy as np
from numba import jit, vectorize, float64, int64

name="sghmc"

@jit([float64[:](float64,float64[:,:],float64[:],float64,float64,float64,float64,int64,int64)],cache=True)
def hmc_1dnb(grad,X,y,theta0,M,C,epsilon,batchsize=50,iter=1000):
    """
    This function outputs the 1 dimiension Hamilton Monte Carlo samples without M-H correction.    
    Args: 
        grad: the gradient function that input X,y,theta and output a tuple with one cooresponding to gradient of log prior
        and another correspongind to log likelihood
        X: the design matrix with row number corresponding to observations and col number corresponding to predictors 
        y: the outcome variable 1-d array
        theta0: the initial point of theta, the parameter of interest
        grad: the gradient of the potential
        M: the mass
        C: the C term, where C*M^{-1} is the friction
        epsilon: stepsize
        batchsize: the number of minibatch used
        iter: iteration number, 1000 by default
    
    Returns:
        out: a 2-d array of joint posterior sample of theta(first column) and momentum(second column)
        
    Examples:
        see odeg
    """

    T=X.shape[0]
    r=np.random.normal(0,np.sqrt(M))
    theta=theta0
    theta_save=np.zeros(iter)
    r_save=np.zeros(iter)
    for t in range(iter):
        theta=theta+epsilon*r/M
        batch=np.random.choice(T,batchsize,replace=False)
        pr,ll=grad(theta,X[batch,:],y[batch])
        r=r-(pr+T*ll)*epsilon-epsilon*C*r/M+np.random.normal(0,np.sqrt(2*epsilon*C))
        theta_save[t]=theta
        r_save[t]=r
    return np.c_[theta_save,r_save]




@jit([float64[:,:](float64[:],float64[:,:],float64[:],float64[:],float64[:,:],float64[:,:],float64, int64,int64)],cache=True)
def hmc_nbvec(grad,X,y,theta0,M,C,epsilon,batchsize=50,iter=1000):
    """
    This function outputs the p dimiension Hamilton Monte Carlo samples without M-H correction.    
    Args: 
        grad: the gradient function that input X,y,theta and output a tuple with one cooresponding to gradient of log prior
        and another correspongind to log likelihood
        X: the design matrix with row number corresponding to observations and col number corresponding to predictors 
        y: the outcome variable 1-d array
        theta0: the initial point of theta, the parameter of interest
        M: the mass
        C: the C term, where C*M^{-1} is the friction
        epsilon: stepsize
        batchsize: the number of minibatch used
        iter: iteration number, 1000 by default
        
    Returns:
        out: a 2p-d array of joint posterior sample of theta(first p columns) and momentum(second p columns)
        
    Examples:
        see hdeg
    """
    n=y.size
    p=theta0.shape[0]
    r=np.random.multivariate_normal(np.zeros(p),M)
    theta=theta0
    theta_save=np.zeros([iter,p])
    r_save=np.zeros([iter,p])
    for t in range(iter):    
        mr=np.linalg.solve(M,r)
        theta=theta+epsilon*mr
        batch=np.random.choice(n,batchsize,replace=False)
        pr,ll=grad(theta,X[batch,:],y[batch])
        r=r-(pr+n*ll)*epsilon-epsilon*np.dot(C,mr)+np.random.multivariate_normal(np.zeros(p),2*epsilon*C,1).ravel()
        theta_save[t,:]=theta
        r_save[t,:]=r
    return np.c_[theta_save,r_save]