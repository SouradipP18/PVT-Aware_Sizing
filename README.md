# PVT-Aware Sizing using INSIGHT

# Target Key innovations:

1. Few-Shot Learning to transfer INSIGHT Nominal (Base Model) to PVT corners
2. Leveraging nominal corner circuit physics and trajectory knowledge as a low-fidelity representation for intelligent targeted sampling in the high-fidelity PVT corners
(Intelligent Sampling)
3. Alternatively we can also use intelligent sampling like explained above, to few-shot learn the other PVT corners, and then use these model predictions as low-fidelity (less accurate) approximations of the actual performance across corners.
4. Attention-based RL using the learnt attention numbers in the Ensemble Models (across PVT) for much better efficiency instead of trial-and-error at the start
