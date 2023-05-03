# Prepare the target model

The blockchain codes are borrowed from [SquirRL](https://github.com/wuwuz/SquirRL). Use the following command to install the dependencies:
```
pip install -r requirements_squirrl.txt
```
Then, run the following command to train and save the blockchain model or you can use the pretrained model in the folder: btc_longest20_100/. Also, you could adjust the eth.py codes to train the ethereum model.
```
python btc_model.py
```

# Generate trajectories

Use the jupyter notebook btc_obs.ipynb to generate the trajectories (Here are some issues with tensorflow 1.14 cuda version on our machine when using the python script, so we use the notebook instead when we would like to load the blockchain model. Also, we recommend copying the target blockchain model since sometimes the noetbook will break the loaded model). Note that due to the length of the trajectories in selfish mining varies a lot, we choose some trajectories with length 10-15 and save their index, since it may not be meaning to explain some trajectories with only 1 or 2 steps. You could adjust the length of the trajectories in the notebook to explain longer trajectories.

# Train the explantion model

With the saved trajectories, we could train the local approxiamtion model. Use the following command to train and save the model:
```
python train_concept.py     #our method
python train_cell.py        #LSTMWithInputCellAttention
python train_attention.py   #Attention
python train_theta.pt       #Linear Regression
```

# Explain the trajectories

Use generate_explain.py and generate_explain_value_based.ipynb to generate the explanation for the trajectories. 

# Fidelity test

Use fidelity_test.ipynb to test the fidelity of the explanation.

# Utility

Use patch.ipynb to run the patch for the target model.