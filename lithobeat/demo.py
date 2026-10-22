"""Reproduce: python -m lithobeat.demo --out examples/generated"""
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import resample_poly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import STFT, Problem, solve, synthetic_profile, fit_diagonal

def beat(sr=16000,seconds=4,seed=20261021):
    rng=np.random.default_rng(seed)
    t=np.arange(int(sr*seconds))/sr
    x=.025*np.sin(2*np.pi*220*t)+.018*np.sin(2*np.pi*330*t)
    for onset in np.arange(.25,seconds-.2,.5):
        u=np.maximum(t-onset,0)
        phase=2*np.pi*(55*u+35*(1-np.exp(-25*u))/25)
        x+=(t>=onset)*(.38*np.sin(phase)*np.exp(-18*u)+.055*rng.normal(size=len(t))*np.exp(-160*u))
    for onset in np.arange(.125,seconds-.1,.25):
        u=np.maximum(t-onset,0)
        x+=(t>=onset)*.025*rng.normal(size=len(t))*np.exp(-90*u)
    return x*.48/abs(x).max()

def run(out):
    out.mkdir(parents=True,exist_ok=True)
    sr=16000;a=STFT(sr*4);x=a.pad(beat());X=a.analysis(x)
    f=np.fft.fftfreq(a.size,1/sr)
    times=(np.arange(X.shape[1])*a.hop+a.size/2)/sr
    onset=sum(np.exp(-((times-q)/.07)**2) for q in np.arange(.25,3.8,.5))
    target=X*(1+.18*np.exp(-(abs(f[:,None])/180)**2)*onset[None,:])
    profiles=np.array([synthetic_profile(f,n) for n in ['club','earbud']])
    masters={};certificates={};cap=.55
    for name,weights in {'club':[1.,0.],'earbud':[0.,1.],'balanced':[.5,.5]}.items():
        p=Problem(a,x,target,profiles,np.array(weights),ridge=.08,cap=cap)
        masters[name],certificates[name]=solve(p)
        if not certificates[name]['converged']:raise RuntimeError('Solver did not converge')
    def error(z,h):
        return float(np.linalg.norm(h[:,None]*a.analysis(z)-target)/np.linalg.norm(target))
    results={};signals={'source':x,**masters}
    for name,z in signals.items():
        results[name]=dict(club_error=error(z,profiles[0]),earbud_error=error(z,profiles[1]),
            processing_relative_l2=float(np.linalg.norm(z-x)/np.linalg.norm(x)),
            rms_dbfs=float(20*np.log10(np.sqrt(np.mean(z*z))+1e-15)),sample_peak=float(abs(z).max()),
            oversampled_4x_peak_diagnostic=float(abs(resample_poly(z,4,1)).max()),
            active_bound_samples=int(np.count_nonzero(abs(z)>=cap-1e-10)))
        wavfile.write(out/f'{name}.wav',sr,z.astype(np.float64))
        rate,readback=wavfile.read(out/f'{name}.wav')
        assert rate==sr and np.array_equal(readback,z) and abs(readback).max()<=cap
        for medium,h in zip(['club','earbud'],profiles):
            wavfile.write(out/f'{name}_through_{medium}.wav',sr,a.adjoint(h[:,None]*a.analysis(z)).astype(np.float64))
    rng=np.random.default_rng(44)
    train=rng.normal(size=X.shape)+1j*rng.normal(size=X.shape)
    test=rng.normal(size=X.shape)+1j*rng.normal(size=X.shape)
    calibration={}
    for name,h in zip(['club','earbud'],profiles):
        fit=fit_diagonal(train,h[:,None]*train+.002*(rng.normal(size=X.shape)+1j*rng.normal(size=X.shape)))
        calibration[name]=float(np.linalg.norm((fit-h)[:,None]*test)/np.linalg.norm(h[:,None]*test))
    sweep=[]
    for ridge in [.01,.08,.5,2.]:
        z,c=solve(Problem(a,x,target,profiles[1:2],np.ones(1),ridge=ridge,cap=cap))
        sweep.append(dict(ridge=ridge,earbud_error=error(z,profiles[1]),processing_relative_l2=float(np.linalg.norm(z-x)/np.linalg.norm(x)),gap_bound=c['gap_upper_bound'],converged=c['converged']))
    report=dict(notice='Synthetic virtual STFT media; not measured speakers/AirPods or listening scores.',
        seed=20261021,sample_rate=sr,samples=len(x),fft_size=a.size,cap=cap,ridge=.08,
        metrics=results,certificates=certificates,ridge_tradeoff=sweep,
        synthetic_calibration_test_relative_error=calibration,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__))
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    plt.rcParams.update({'figure.facecolor':'#101827','axes.facecolor':'#101827','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#566274'})
    keep=(f>0)&(f<=7000)
    fig,axs=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for name,h in zip(['Club simulation','Earbud simulation'],profiles):
        axs[0,0].semilogx(f[keep],20*np.log10(h[keep]),label=name)
    axs[0,0].set(title='Invented playback models',xlabel='Frequency (Hz)',ylabel='Response (dB)');axs[0,0].legend()
    labels=list(results)
    for i,key in enumerate(['club_error','earbud_error']):
        axs[0,1].bar(np.arange(4)+i*.36,[results[n][key] for n in labels],width=.36,label=key.replace('_',' '))
    axs[0,1].set_xticks(np.arange(4)+.18,labels)
    axs[0,1].set(title='Each medium rewards a different master',ylabel='Relative complex STFT error (lower is better)');axs[0,1].legend()
    for name,c in certificates.items():
        hist=np.array(c['history']);axs[1,0].semilogy(hist[:,0],np.maximum(hist[:,2],1e-16),label=name)
    axs[1,0].set(title='Computed optimality gap bounds',xlabel='Iterations',ylabel='Upper bound on objective gap');axs[1,0].legend()
    for name,z in signals.items():axs[1,1].plot(np.arange(len(z))/sr,z,alpha=.65,label=name)
    for bound in [-cap,cap]:axs[1,1].axhline(bound,color='#ffb454',ls='--')
    axs[1,1].set(xlim=(.24,.36),title='Sample headroom is part of the solve',xlabel='Time (s)',ylabel='Amplitude');axs[1,1].legend()
    fig.suptitle('LithoBeat • optimize what the medium reproduces',fontsize=20)
    fig.savefig(out/'overview.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(name,z) in zip(axs,[('Source',x),('Club master',masters['club']),('Earbud master',masters['earbud'])]):
        im=ax.pcolormesh(times,f[keep],20*np.log10(abs(a.analysis(z)[keep])+1e-6),vmin=-65,vmax=5,shading='auto',cmap='magma')
        ax.set(title=name,ylim=(30,4000),yscale='log',xlabel='Time (s)',ylabel='Frequency (Hz)')
    fig.colorbar(im,ax=axs,label='STFT coefficient magnitude (dB)');fig.savefig(out/'spectrograms.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(6.75,12),layout='constrained')
    for name,h in zip(['Club','Earbud'],profiles):axs[0].semilogx(f[keep],20*np.log10(h[keep]),label=name)
    axs[0].set(title='SAME BEAT. DIFFERENT MEDIUM.',ylabel='Simulated response (dB)');axs[0].legend()
    axs[1].bar(['Source','Club master','Earbud master'],[results[n]['earbud_error'] for n in ['source','club','earbud']],color=['#9aa5b8','#6bdac5','#ffb454'])
    axs[1].set(title='Optimize the reproduced result',ylabel='Earbud-model error ↓')
    axs[2].imshow(20*np.log10(abs(a.analysis(masters['earbud'])[keep])+1e-6),aspect='auto',origin='lower',cmap='magma',vmin=-65,vmax=5)
    axs[2].set(title='A 2D representation, a constrained waveform',xlabel='Time frame',ylabel='Frequency bin')
    fig.suptitle('LithoBeat',fontsize=32)
    fig.get_layout_engine().set(rect=(0,.035,1,.965))
    fig.text(.5,.005,'Synthetic media • no hardware measurements • not a listening test',ha='center',fontsize=9)
    fig.savefig(out/'tiktok_portrait.png',dpi=160);plt.close(fig)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file() and p.name!='sha256.json'}
    (out/'sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('examples/generated'))
    print(json.dumps(run(p.parse_args().out)['metrics'],indent=2))
if __name__=='__main__':main()
