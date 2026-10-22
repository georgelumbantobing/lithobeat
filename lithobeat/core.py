"""Constrained inverse-medium mastering in a Parseval STFT frame."""
from dataclasses import dataclass
import numpy as np
from scipy.fft import fft, ifft

class STFT:
    """Periodic sqrt-Hann half-overlap full FFT: A* A = I."""
    def __init__(self, samples, size=512):
        if samples < 1 or size < 4 or size % 2:
            raise ValueError("Positive length and even FFT size >=4 required")
        self.size, self.hop = size, size//2
        self.n = max(size, ((samples+self.hop-1)//self.hop)*self.hop)
        self.indices=(np.arange(0,self.n,self.hop)[:,None]+np.arange(size))%self.n
        self.window=np.sin(np.pi*np.arange(size)/size)
    def pad(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=1 or not len(x) or len(x)>self.n or not np.all(np.isfinite(x)):
            raise ValueError("Finite nonempty mono waveform required")
        return np.pad(x,(0,self.n-len(x)))
    def analysis(self,x):
        return fft(x[self.indices]*self.window,norm="ortho",axis=1).T
    def adjoint(self,z):
        frames=ifft(z.T,norm="ortho",axis=1).real*self.window
        out=np.zeros(self.n)
        np.add.at(out,self.indices.ravel(),frames.ravel())
        return out

@dataclass
class Problem:
    transform: STFT
    source: np.ndarray
    target: np.ndarray
    profiles: np.ndarray
    weights: np.ndarray
    ridge: float=.08
    cap: float=.55
    def __post_init__(self):
        self.source=np.asarray(self.source,dtype=float)
        self.target=np.asarray(self.target,dtype=complex)
        self.profiles=np.asarray(self.profiles,dtype=complex)
        self.weights=np.asarray(self.weights,dtype=float)
        shape=self.transform.analysis(np.zeros(self.transform.n)).shape
        if (self.source.shape!=(self.transform.n,) or self.target.shape!=shape
            or self.profiles.ndim!=2 or self.profiles.shape[1]!=shape[0]
            or not len(self.profiles) or self.weights.shape!=(len(self.profiles),)):
            raise ValueError("Incompatible dimensions")
        if not all(np.all(np.isfinite(a)) for a in [self.source,self.target,self.profiles,self.weights]):
            raise ValueError("Finite arrays required")
        if (not np.isfinite(self.ridge) or self.ridge<=0 or not np.isfinite(self.cap)
            or not 0<self.cap<=1 or np.any(self.weights<0) or not np.isclose(self.weights.sum(),1)):
            raise ValueError("Positive ridge, cap in (0,1], probability weights required")
        self.power=np.sum(self.weights[:,None]*abs(self.profiles)**2,axis=0)[:,None]
        self.drive=np.sum(self.weights[:,None]*self.profiles.conj(),axis=0)[:,None]*self.target
        self.lipschitz=float(self.power.max()+self.ridge)
    def value_gradient(self,z):
        a=self.transform.analysis(z)
        value=.5*sum(w*np.sum(abs(h[:,None]*a-self.target)**2) for w,h in zip(self.weights,self.profiles))
        value+=.5*self.ridge*np.sum((z-self.source)**2)
        g=self.transform.adjoint(self.power*a-self.drive)+self.ridge*(z-self.source)
        return float(value),g
    def gap_bound(self,z,g):
        d=np.clip(z-g/self.ridge,-self.cap,self.cap)-z
        return float(max(0.,-np.dot(g,d)-.5*self.ridge*np.dot(d,d)))

def solve(problem,tolerance=1e-8,max_iterations=3000):
    if tolerance<=0 or max_iterations<1:
        raise ValueError("Positive tolerance and budget required")
    z=np.clip(problem.source,-problem.cap,problem.cap)
    initial,_=problem.value_gradient(z)
    history=[]
    for iteration in range(max_iterations+1):
        value,g=problem.value_gradient(z)
        gap=problem.gap_bound(z,g)
        history.append((iteration,value,gap))
        if gap<=tolerance*max(1.,initial):
            break
        if iteration<max_iterations:
            z=np.clip(z-g/problem.lipschitz,-problem.cap,problem.cap)
    return z,dict(iterations=iteration,objective=value,gap_upper_bound=gap,
        relative_gap_bound=gap/max(1.,initial),converged=gap<=tolerance*max(1.,initial),
        sample_peak=float(abs(z).max()),history=history)

def synthetic_profile(frequencies,kind,bass_db=0.):
    points=[0,35,65,120,500,3000,8000]
    if kind=="club":
        gains=[-22,-10,3,4,0,-2,-5]
    elif kind=="earbud":
        gains=[-30,-20,-12,-6,0,3,-4]
    else:
        raise ValueError("Unknown synthetic profile")
    f=abs(frequencies)
    return 10**((np.interp(f,points,gains)+bass_db*np.exp(-(f/180)**2))/20)

def fit_diagonal(reference_coefficients,observed_coefficients,ridge=1e-8):
    """Ridge fit on aligned training pairs; validate on separate recordings."""
    x,y=np.asarray(reference_coefficients),np.asarray(observed_coefficients)
    if x.shape!=y.shape or x.ndim!=2 or not np.isfinite(ridge) or ridge<=0:
        raise ValueError("Matching 2D coefficients and positive ridge required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Finite calibration coefficients required")
    return np.sum(x.conj()*y,axis=1)/(np.sum(abs(x)**2,axis=1)+ridge)
