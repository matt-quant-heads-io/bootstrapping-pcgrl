source ./set_proj_root.sh
conda activate expert_traj
echo "activated conda env expert_traj..."

RDMAV_FORK_SAFE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python gen_expert_traj.py  
