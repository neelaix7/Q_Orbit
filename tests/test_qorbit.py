import numpy as np
from src.simulator.lightcurve_generator import generate_single_light_curve
from src.classical.feature_engineering import extract_all_features
from src.features.physics_informed import extract_physics_informed
from src.quantum.vqc import PureVQC
from src.quantum.hybrid_model import HybridQuantumClassifier
from src.evaluation.degradation import add_gaussian_noise, truncate_observation, add_missing
from src.fusion.signal_quality import analyze_signal_quality
from src.fusion.adaptive_fusion import AdaptiveFusionGate
import torch

def test_dataset():
    c,_=generate_single_light_curve(0, n_samples=256)
    assert c.shape==(256,)
    assert 0<=c.min() and c.max()<=1

def test_features():
    c=np.random.rand(256)
    f=extract_all_features(c)
    assert f.shape==(21,)
    pi=extract_physics_informed(c)
    assert pi['vector'].shape==(21,)

def test_quantum_dims():
    m=PureVQC(n_qubits=8, n_layers=2)
    x=torch.randn(4,8)
    logits, proba=m(x)
    assert logits.shape==(4,5)
    assert torch.allclose(proba.sum(1), torch.ones(4), atol=1e-5)

def test_hybrid():
    m=HybridQuantumClassifier(n_features=21, n_qubits=8)
    x=torch.randn(4,21)
    logits, proba=m(x)
    assert logits.shape==(4,5)

def test_degradation():
    cur=np.random.rand(10,256)
    noisy=add_gaussian_noise(cur,0.1)
    assert noisy.shape==cur.shape
    trunc=truncate_observation(cur,0.5)
    assert trunc.shape==cur.shape
    miss=add_missing(cur,0.1)
    assert miss.shape==cur.shape

def test_fusion():
    g=AdaptiveFusionGate(d_classical=5,d_quantum=5,d_quality=7)
    pc=torch.randn(4,5); pq=torch.randn(4,5); q=torch.randn(4,7)
    alpha=g(pc,pq,q)
    assert alpha.shape==(4,1)
    assert 0<alpha.min() and alpha.max()<1

def test_signal():
    c=np.random.rand(256)
    sq=analyze_signal_quality(c)
    assert 0<=sq['signal_quality_score']<=1
    assert sq['quality_label'] in ['Low','Medium','High']

if __name__=='__main__':
    test_dataset(); test_features(); test_quantum_dims(); test_hybrid(); test_degradation(); test_fusion(); test_signal()
    print('All tests passed')
