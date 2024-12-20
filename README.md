# bootstrapping-pcgrl
## Install both environments (python 3.5 for training tensorflow models from original pcgrl, and python 3.10 for torch models & stable-baselines3)
```
sh scripts/install.sh
```

# Run training (tensorflow runtime)
```
conda activate bs_pcgrl_py_35
python train.py 
```