# Quantum Light-Curve Forensics - Evaluation Report

**Generated**: 2026-08-17 15:37:38
**Test samples**: 2000
**Random seed**: 42

## Comparison Summary

| Model | Accuracy | F1-Macro |
|-------|----------|----------|
| Hybrid Quantum | 68.86% | 0.6832 |
| Classical CNN | 87.01% | 0.8695 |
| Classical SVM | 74.96% | 0.7481 |

## Per-Class Detailed Results

| Class | Quantum F1 | CNN F1 | SVM F1 |
|-------|------------|---------|--------|
| Intact Satellite | 0.8182 | 1.0000 | 0.8374 |
| Dead Satellite | 0.8256 | 1.0000 | 0.8456 |
| Rocket Body | 0.4902 | 0.6963 | 0.6199 |
| Fragmentation Debris | 0.7807 | 1.0000 | 0.8397 |
| Spoofed Satellite | 0.5013 | 0.6512 | 0.5977 |

## Key Findings

- Best performing model: **Classical CNN** with F1-macro of 0.8695
- Quantum hybrid vs classical baselines: comparison shown above
- Difficulty sweep at varying noise levels recommended for further analysis
- Per-class analysis: see confusion matrices and per-class F1 scores above
