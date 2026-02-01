# D2SRS: Dense-to-Sparse Reward Switching for AGV Task Allocation

This repository contains the official implementation of the paper:

**An Order-Oriented AGV Task Allocation Method Based on Multi-Agent Reinforcement Learning with a Dual-Reward Strategy**

> Yike Shi, Shengqi Lai, Siyuan Jin, Yuanjun Laili  

---

## 🔍 Overview

This work proposes **D2SRS (Dense-to-Sparse Reward Switching)**, a multi-agent reinforcement learning algorithm for **order-oriented AGV task allocation in intelligent workshops**.

The key idea is to:
- Use **dense rewards** in early training stages to accelerate convergence
- **Dynamically switch** to **sparse rewards** to improve global exploration and avoid local optima
- Build on the **Independent Proximal Policy Optimization (IPPO)** framework
- Coordinate **task allocation agents** and **AGV execution agents** via shared global objectives

Experimental results show that D2SRS:
- Increases the number of orders completed on time by **at least 7.77%**
- Reduces average processing time per order by **13.2%**
compared with heuristic methods and fixed-reward MARL baselines.

## How to Run

Main algorithm (D2SRS):

python IPPO_main_D2SPR.py

Baseline experiments:

Dense reward only:
python IPPO_main_Dense_only.py

Sparse reward only:
python IPPO_main_Sparse_only.py

--------------------------------------------------

## Algorithm Description

The D2SRS algorithm consists of three phases:

1. Dense Reward Guidance Phase  
   Dense rewards are used to guide early-stage learning, including task allocation,
   task execution, and order completion rewards.

2. Reward Switching Mechanism  
   When the reward change rate converges or the iteration threshold is reached,
   the reward function switches from dense to sparse, and historical experiences
   are re-evaluated.

3. Sparse Reward Optimization Phase  
   Sparse rewards based on the number of orders completed on time are used to
   encourage long-term global optimization.

The algorithm is implemented under the Independent PPO framework with cooperative agents.

--------------------------------------------------

## Experimental Results

Experiments are conducted in a simulated intelligent workshop environment with
multiple production lines, assembly stations, and AGVs.

The proposed D2SRS algorithm is compared with:
- FCFS (First-Come-First-Served)
- LCFS (Last-Come-First-Served)
- HPR (Highest Priority Rule)
- RAND (Random)

Results show that D2SRS outperforms heuristic and fixed-reward baselines in both
the number of orders completed on time and the average task completion time.

--------------------------------------------------

## Citation

If you use this code in your research, please cite:

@article{Shi2024D2SRS,
  title = {An Order-Oriented AGV Task Allocation Method Based on Multi-Agent Reinforcement Learning with a Dual-Reward Strategy},
  author = {Shi, Yike and Lai, Shengqi and Jin, Siyuan and Laili, Yuanjun},
  year = {2024}
}

--------------------------------------------------

## Contact

Yike Shi  
Email: syk1529795850@gmail.com
