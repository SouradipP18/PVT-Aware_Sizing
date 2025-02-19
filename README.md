# PVT-Aware Sizing using INSIGHT

Target Key innovations:
1. Few-Shot Learning to transfer INSIGHT Nominal (Base Model) to PVT corners
2. With Knowledge from Nominal corner about the Circuit Physics and trajectory, treating Nominal as a Low-Fidelity Design Space for sampling from the PVT corners as High-Fidelity (Intelligent Sampling) 
3. Attention-based RL using the learnt attention numbers in the Ensemble Models (across PVT) for much better efficiency instead of trial-and-error at the start
