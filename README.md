# Safe Curriculum Generation for Constrained RL

Codebase for ICLR 2025 paper [_Safety-Prioritizing Curricula for Constrained Reinforcement Learning_](https://openreview.net/forum?id=f3QR9TEERH&referrer=%5Bthe%20profile%20of%20Cevahir%20Koprulu%5D(%2Fprofile%3Fid%3D~Cevahir_Koprulu1)) by Cevahir Koprulu, Thiago D. Simão, Nils Jansen, and Ufuk Topcu.

Our codebase is built on the repository of _Curriculum Reinforcement Learning via Constrained Optimal Transport_ (CURROT) by Klink et al. (2022).

Web sources for CURROT:

Source code: https://github.com/psclklnk/currot/tree/icml (ICML branch)

Paper: https://proceedings.mlr.press/v162/klink22a.html

We use the constrained RL algorithms implemented in OmniSafe:

Web sources for OmniSafe:

Source code: https://www.omnisafe.ai/en/latest/index.html#


We run our codebase on Ubuntu 20.04.5 LTS with Python 3.10

## Installation

The required packages are provided below with the bash commands for installation:
```bash
conda install -c conda-forge cyipopt
pip install torch torchvision torchaudio
pip install omnisafe
pip install scikit-learn
pip install gpytorch
pip install pyro-ppl
pip install gurobipy
pip install geomloss
```

## How to run
To run a single experiment (training + evaluation), *run.py* can be called as follows (you can put additional parameters):
```bash
python run.py --train --eval --env safety_maze_3d --type constrained_wasserstein --learner PPOLag --seed 1 # SCG
python run.py --train --eval --env safety_maze_3d --type wasserstein --learner PPOLag --PEN_COEFT 1.0 --seed 1 # NaiveSafeCURROT
python run.py --train --eval --env safety_maze_3d --type wasserstein --learner PPOLag --PEN_COEFT 0.0 --seed 1 # CURROT
python run.py --train --eval --env safety_maze_3d --type wasserstein4cost --learner PPOLag --seed 1 # CURROT4Cost
python run.py --train --eval --env safety_maze_3d --type self_paced --learner PPOLag --seed 1 # SPDL
python run.py --train --eval --env safety_maze_3d --type plr --learner PPOLag --seed 1 # PLR
python run.py --train --eval --env safety_maze_3d --type alp_gmm --learner PPOLag --seed 1 # ALP-GMM
python run.py --train --eval --env safety_maze_3d --type goal_gan --learner PPOLag --seed 1 # GoalGAN
python run.py --train --eval --env safety_maze_3d --type default --learner PPOLag --seed 1 # Default
```

## Evaluation
Under *misc* directory, you can the scripts to obtain figures in the paper:
1) *plot_progression_of_results.py*: We use this script to plot the progression of reward, success, cost, regret during training and in contexts from target context distributions.
2) *plot_final_stats.py*: We run this script to obtain a box plots for expected success, discounted cumulative reward and cost, as well as regret, obtained at the final curriculum iteration.
3) *plot_safety_maze_curriculum_progression.py*: We run this script to obtain a curriculum evolution figure in safety-maze.
4) *plot_safety_goal_curriculum_progression.py*: We run this script to obtain a curriculum evolution figure in safety-goal.
5) *plot_safety_passage_curriculum_progression.py*: We run this script to obtain a curriculum evolution figure in safety-passage.
6) *plot_safety_passage_push_curriculum_progression.py*: We run this script to obtain a curriculum evolution figure in safety-push.
7) *sample_eval_contexts.py*: We run this script to draw contexts from the target context distributions and record them to be used for evaluation of trained policies.



