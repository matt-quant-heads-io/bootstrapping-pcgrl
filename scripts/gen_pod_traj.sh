source ./set_proj_root.sh
conda activate bs_pcgrl_py_310
echo "activated conda env bs_pcgrl_py_310..."

python gen_expert_traj.py
