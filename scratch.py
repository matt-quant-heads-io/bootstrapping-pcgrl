import numpy as np
import os
# Create a 1D array of strings

data_dir = "/home/jupyter-msiper/bootstrapping-pcgrl/goal_maps/expert_zelda"
write_dir = "/home/jupyter-msiper/bootstrapping-pcgrl/goal_maps/zelda_expert"


for idx, file in enumerate(os.listdir(data_dir)):
    print(f"{file}")
    lvl_data = None
    with open(f"{data_dir}/{file}", "r") as f:
        lvl_data = f.read()

    lvl_arr = np.array([c for c in lvl_data])
    lvl_arr = lvl_arr.reshape(7, 11)

    with open("{write_dir}/{idx}.txt".format(write_dir=write_dir, idx=str(idx)), "w") as f:
        for row in lvl_arr:
            f.write(''.join(row)+"\n")

    