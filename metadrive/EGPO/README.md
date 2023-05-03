# Prepare the target model

The blockchain codes are borrowed from [EGPO](https://github.com/decisionforce/EGPO). Follow the instructions to install the requirements(the training scripts are included in this folder). Use the following command to prepare the target model:
```
python dagger_model.py
```
Note that this is the offline reinforcement learning method and does not directly estimate a value function where the value function-based explanation methods could be applied  directly. To evaluate safe driving with value-based method, you can estimate the value function from the trained policy since the expert's behavior is assumed to be optimal, or you can use DQN to train the agent, or combine online learning with offline learning ([JSRL](https://arxiv.org/pdf/2204.02372.pdf&) or [Kickstarting](https://arxiv.org/pdf/1803.03835.pdf)).


To evaluate the safe driving in a fixed environment, we fix the seed to make sure the environment is the same for different trajectories in the training scripts.

# Generate trajectories

Run dagger_obs.py to generate the trajectories. 

# Train the explantion model

With the saved trajectories, we could train the local approxiamtion model. Use the following command to train and save the model:
```
python train_concept.py     #our method
python train_cell.py        #LSTMWithInputCellAttention
python train_attention.py   #Attention
python train_theta.pt       #Linear Regression
```

# Explain the trajectories

Use generate_explain.py to generate the explanation for the trajectories. 

# Fidelity test

Run run_fidelity.sh to test the fidelity of the explanation.

# Utility

Run dagger_retrain.py to retrain the target model with the explanation and dagger_evaluate.py to evaluate the retrained model.