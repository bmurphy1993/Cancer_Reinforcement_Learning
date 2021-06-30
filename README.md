# Cancer_Reinforcement_Learning

RL_simulation.py generates baseline results from the unrestricted DQN agent and updates Table_01.xlsx

RL_simulation_res.py generates results from the unrestricted DQN agent

RL_simulation_nccn.py generates results from the NCCN agent

RL_simulation_sparse.py generates results from the restricted DQN agent with sparse rewards. Sparse rewards are equivalent to standard rewards for an entire patient trajectory, but are administed to agents only at the end of a full round, rather than incrementally after each state.

"Figures" notebooks generate report figures for the unrestricted DQN, restricted DQN, and sparse rewards versions of the analysis.

Steps to run the cancer reinforcement learning simulation and generate figures:
  1. Run RL_simulation_res.py for baseline DQN results, RL_simulation_res.py for restricted DQN, and RL_simulation_nccn.py for the NCCN agent results.
  2. Once python scripts finish running, open "Figures" notebooks in Jupyter to generate report figures.
  3. Repeat steps with "sparse" files if desired, though this version of the experiment is under development and results are not included in any analysis.

Notes
  - All simulations are set to run for 200,000 rounds by default
  - Number of rounds and all other hyperparameters can be adjusted in "RL_simulation" files
  - Report figures will export to "figs" folder
