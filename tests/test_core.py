import unittest
import numpy as np
from scipy.optimize import minimize
from scipy.io import wavfile
from pathlib import Path
from lithobeat import STFT, Problem, solve, synthetic_profile, fit_diagonal

class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(7);self.a=STFT(32,8)
        self.x=self.rng.normal(size=32)*.15
        self.t=self.a.analysis(self.x)*1.3
    def problem(self,h=None,cap=.18,ridge=.2):
        if h is None:h=np.linspace(.2,1.4,8)
        return Problem(self.a,self.x,self.t,np.array([h]),np.ones(1),ridge,cap)
    def test_parseval_and_adjoint(self):
        for length in [1,7,31,32,33]:
            a=STFT(length,8);x=a.pad(self.rng.normal(size=length))
            np.testing.assert_allclose(a.adjoint(a.analysis(x)),x,atol=1e-14)
            self.assertAlmostEqual(np.sum(abs(a.analysis(x))**2),np.dot(x,x),places=12)
            z=self.rng.normal(size=a.analysis(x).shape)+1j*self.rng.normal(size=a.analysis(x).shape)
            self.assertAlmostEqual(np.vdot(a.analysis(x),z).real,np.dot(x,a.adjoint(z)),places=12)
    def test_gradient_finite_difference(self):
        p=self.problem();z=self.rng.normal(size=32);d=self.rng.normal(size=32);e=1e-6
        _,g=p.value_gradient(z)
        fd=(p.value_gradient(z+e*d)[0]-p.value_gradient(z-e*d)[0])/(2*e)
        self.assertAlmostEqual(fd,np.dot(g,d),places=7)
    def test_independent_optimizer_and_gap(self):
        p=self.problem();z,c=solve(p,tolerance=1e-12)
        reference=minimize(p.value_gradient,np.zeros(32),jac=True,method='L-BFGS-B',bounds=[(-p.cap,p.cap)]*32,options={'ftol':1e-15,'gtol':1e-10})
        np.testing.assert_allclose(z,reference.x,atol=2e-6)
        self.assertLessEqual(c['objective']-reference.fun,c['gap_upper_bound']+1e-10)
        self.assertTrue(c['converged']);self.assertLessEqual(abs(z).max(),p.cap)
    def test_identity_closed_form(self):
        p=self.problem(np.ones(8));z,c=solve(p,tolerance=1e-13)
        expected=np.clip((1.3+self.problem().ridge)/(1+self.problem().ridge)*self.x,-p.cap,p.cap)
        np.testing.assert_allclose(z,expected,atol=1e-10)
    def test_monotone_and_feasible(self):
        p=self.problem();z,c=solve(p)
        self.assertTrue(np.all(np.diff(np.array(c['history'])[:,1])<=1e-12))
        self.assertGreater(np.count_nonzero(abs(z)>p.cap-1e-8),0)
    def test_zero_and_null_medium(self):
        for h in [np.zeros(8),np.ones(8)]:
            p=Problem(self.a,np.zeros(32),np.zeros_like(self.t),np.array([h]),np.ones(1))
            z,c=solve(p);np.testing.assert_array_equal(z,np.zeros(32));self.assertTrue(c['converged'])
    def test_fit_held_out(self):
        h=synthetic_profile(np.fft.fftfreq(8,1/16000),'earbud')
        x=self.rng.normal(size=(8,200))+1j*self.rng.normal(size=(8,200))
        learned=fit_diagonal(x,h[:,None]*x)
        np.testing.assert_allclose(learned,h,atol=1e-9)
    def test_validation_and_iteration_exhaustion(self):
        for kwargs in [{'ridge':0},{'cap':0},{'ridge':float('nan')}]:
            with self.assertRaises(ValueError):self.problem(**kwargs)
        with self.assertRaises(ValueError):STFT(0)
        with self.assertRaises(ValueError):self.a.pad([float('nan')])
        z,c=solve(self.problem(),max_iterations=1,tolerance=1e-30)
        self.assertFalse(c['converged']);self.assertLessEqual(abs(z).max(),.18)
    def test_exported_demo(self):
        folder=Path('examples/generated')
        if not folder.exists():self.skipTest('Run demo first')
        import json
        r=json.loads((folder/'results.json').read_text())
        for name in ['source','club','earbud','balanced']:
            sr,z=wavfile.read(folder/f'{name}.wav')
            self.assertEqual(sr,r['sample_rate']);self.assertLessEqual(abs(z).max(),r['cap'])
        for name in ['club','earbud']:
            self.assertLess(r['metrics'][name][name+'_error'],r['metrics']['source'][name+'_error'])
        for c in r['certificates'].values():self.assertTrue(c['converged'])

if __name__=='__main__':unittest.main()
