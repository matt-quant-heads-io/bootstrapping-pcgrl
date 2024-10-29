source ./set_proj_root.sh
conda activate sb2
echo "activated conda env sb2..."
# RDMAV_FORK_SAFE=1 python pipeline.py --train_process bootstrap_then_ppo --goals_for_traj_gen standard  --num_bootstrap_epochs 50 --num_ppo_timesteps 5000 --num_boostrap_episodes 50 --domain zelda   
# RDMAV_FORK_SAFE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python pipeline.py --train_process ppo --goals_for_traj_gen standard  --num_bootstrap_epochs 100 --num_ppo_timesteps 1000000 --num_boostrap_episodes 5000 --domain zelda   

# RDMAV_FORK_SAFE=1 python pipeline.py --resume --train_process bootstrap_then_ppo --goals_for_traj_gen standard  --num_bootstrap_epochs 100 --num_ppo_timesteps 1000000 --num_boostrap_episodes 10000 --domain zelda
RDMAV_FORK_SAFE=1 python pipeline.py --resume --train_process bootstrap_then_ppo --goals_for_traj_gen standard  --num_bootstrap_epochs 100 --num_ppo_timesteps 1000000 --num_boostrap_episodes 10000 --domain zelda