# Prepare the target model

The CyberBattleSim agent model codes are borrowed from [CyberBattleSim](https://github.com/microsoft/CyberBattleSim). To make it easy to modify the game environment, we include the codes in this folder and turn the CyberBattleSim python package to local codes that can be imported, so you do not need to create the docker as the official document guides. Note that this is a history version of the CyberBattleSim and the latest version may not be compatible with our codes. Also, we did some modifications to the source codes to generate the explanation. 

Run chain_model.py to prepare the target model. In our work, we used the chain model, you can switch to another environment by changing the environment name in the code. Also, you can change the ./simulation/generate_network.py to adjust the network topology. You can use plot.ipynb to see the network topology.

# Generate trajectories
Run chain_obs.py to generate the trajectories.

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
Run fidelity_test.py to test the fidelity of the explanation.
