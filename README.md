# InverseRL-Pytorch

## Agent Algorithm
* PPO

## Discriminator Algorithm
* GAIL(Generative Adversarial Imitation Learning)
* VAIL(Variational Adversarial Imitation Learning)
* AIRL(Adversarial Inverse Reinforcement Learning)
  * Two value functions must be merged into one.
* VAIRL(Variational Adversarial Inverse Reinforcement Learning)
  * Joint gaussian distribution kl-divergence yet.
* EAIRL(Empowerment-regularized Adversarial Inverse Reinforcement Learning)
  * Two value functions must be merged into one.
  * Target empowerment must be updated in policy learning step. how?
  * It shows sudden divergence problem.
## TODO
* add SAC
* add SQIL
* add more environments(ant and disabled ant)
* build setup file
* make expert
* make trajectories by expert
