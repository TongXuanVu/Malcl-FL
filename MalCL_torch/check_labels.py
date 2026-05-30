import torch
import os

centralized_dir = r"C:\FederatedLearning\FL\core\data_split\centralized_data"
for i in range(1, 7):
    path = os.path.join(centralized_dir, f"centralized_task_{i}.pt")
    if os.path.exists(path):
        d = torch.load(path, map_location='cpu')
        y = d['y']
        print(f"Task {i}: min={y.min()}, max={y.max()}, unique={len(torch.unique(y))}")
    else:
        print(f"Task {i} not found")
