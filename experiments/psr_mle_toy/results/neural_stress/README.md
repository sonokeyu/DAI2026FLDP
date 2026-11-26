# Neural POMDP stress test

Latent state space: 256; horizon: 8; actions: 4. An autoregressive GRU is fit by trajectory conditional MLE. Evaluation reports next-observation NLL on held-out uniform and action-shifted policies. This is an empirical prediction stress test; it does not compute the paper's exact all-policy TV bonus certificate.
