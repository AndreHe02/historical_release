This is the codebase for Neural Unsupervised Reconstruction of Protolanguage Word Forms (ACL 2023).
A rough outline of the file contents:

- `sampler.py` runs baselines and re-implementations of models from previous work
- `lstm_model.py` runs our neural reconstruction method
- `dp.py` implements the forward-backward dynamic program
- `proposal.py` implements the proposal heuristic we use during Metropolis-Hastings 
- `data.py` contains data preprocessing code
- `lstm.sh`, `lstm2.sh`, and `baselines.sh` contains commands used to run experiments and baselines

