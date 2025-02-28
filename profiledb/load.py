import numpy as np
import os

model = 'llama2_13B'
# base_path=os.path.join('.',dir_path)
base_path=os.path.join('.')
model = [model]
tp = [1,2,4]
gpu_type = ['A10','A6000']
framework_type = ['_ds']
for m in model:
    for g in gpu_type:
        for t in tp:
            for f in framework_type:
                file_name = f"{base_path}/{m}_{g}_{t}{f}.npy"
                data_1 = np.load(file_name)
                print(f"{file_name}, len: {len(data_1)}")
                print(f"{data_1[0]*1000:.2f}, {data_1[1]*1000:.2f}, {data_1[-1]*1000:.2f}")