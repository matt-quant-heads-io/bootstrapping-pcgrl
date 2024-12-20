# Installs stable_baselines2 for training original pcgrl (tf) models
source ./set_proj_root.sh
cd $PROJECT_ROOT
conda create -n bs_pcgrl_py_35 python=3.5 -y
conda activate bs_pcgrl_py_35
pip install -r ${PROJECT_ROOT}/env_requirements/bs_pcgrl_py_35.txt 
pip install -e .
conda deactivate bs_pcgrl_py_35

# Installs stable_baselines3 and torch
conda create -n bs_pcgrl_py_310 python=3.10 -y
conda activate bs_pcgrl_py_310
pip install -e .
pip install -r ${PROJECT_ROOT}/env_requirements/bs_pcgrl_py_310.txt 
conda deactivate bs_pcgrl_py_310
