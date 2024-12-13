import os
import pickle
import pdb
import numpy as np
from plot_xyz import polt_fn

# Specify the directory containing the .pkl files
directory = "/store/real/shuang/mocap_umi"


# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".pkl"):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'rb') as file:
            print(f"Loading {filepath} {filename}")
            data = pickle.load(file)
            
            init_time = data['umi_gripper'][0][1]
            # pdb.set_trace()
            
            posrot = [tem[0] for tem in data['umi_gripper']]
            posrot = np.stack(posrot, axis=0)
            timestamps = [tem[1]-init_time for tem in data['umi_gripper']]
            os.makedirs('plot_align', exist_ok=True)
            polt_fn(timestamps, posrot[:,0], posrot[:,1], posrot[:,2], 'plot_align/'+filename[:-4])
            
            
            