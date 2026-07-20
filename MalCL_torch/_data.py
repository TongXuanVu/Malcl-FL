import torch
import numpy as np
import os

def get_global_label_map(data_root=None):
    """
    Bo data 100-client giu NGUYEN label ID goc voi thu tu task phi tuan tu
    (`task_mapping_label_ids.json`). Code CIL gia dinh label tuan tu 0..33 theo thu tu task,
    nen tra ve dict {label_goc: label_tuan_tu}.

    Bo data cu (da tuan tu san) khong co file json -> tra ve None (khong remap, tuong thich nguoc).
    """
    import json, glob

    candidates = []
    if data_root:
        candidates += [
            os.path.join(data_root, "task_mapping_label_ids.json"),
            os.path.join(os.path.dirname(data_root), "task_mapping_label_ids.json"),
        ]
    if os.path.exists("/kaggle/input"):
        candidates += sorted(glob.glob("/kaggle/input/**/task_mapping_label_ids.json", recursive=True))
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "task_mapping_label_ids.json"))

    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                task_orders = json.load(f)
            flat = [int(c) for task in task_orders for c in task]
            if sorted(flat) != list(range(len(flat))):
                print(f"[MalCL-FL] CANH BAO: {path} khong phu kin 0..N-1, bo qua remap.")
                continue
            mapping = {orig: seq for seq, orig in enumerate(flat)}
            print(f"[MalCL-FL] Remap label goc -> tuan tu theo: {path}")
            return mapping

    print("[MalCL-FL] Khong thay task_mapping_label_ids.json -> gia dinh label da tuan tu.")
    return None

def get_client_data(config, client_id, task_id, global_label_map=None):
    federated_dir = config.train_data if hasattr(config, 'train_data') and config.train_data else r"c:\FederatedLearning\FL\core\data_split\federated_data"
    # Note: task_id in federated_data is 1-indexed (1 to 6)
    path = os.path.join(federated_dir, f"client_{client_id}_task_{task_id}.pt")
    
    if not os.path.exists(path):
        return None, None
        
    data = torch.load(path, map_location='cpu', weights_only=False)
    x = data['x'].numpy()
    y = data['y'].numpy()
    
    if config.debug:
        limit = min(500, len(x))
        indices = np.random.choice(len(x), limit, replace=False)
        x = x[indices]
        y = y[indices]
        
    if global_label_map is not None:
        y = np.array([global_label_map.get(l, l) for l in y])
        
    return x, y

def get_global_test_data(config, global_label_map=None):
    test_path = config.test_data if hasattr(config, 'test_data') and config.test_data else r"c:\FederatedLearning\FL\core\data_split\global_test_data.pt"
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Global Test Set not found at {test_path}.")
        
    test_data = torch.load(test_path, map_location='cpu', weights_only=False)
    if isinstance(test_data, dict):
        X_test, Y_test = test_data['x'], test_data['y']
    else:
        X_test, Y_test = test_data
        
    X_test = X_test.numpy()
    Y_test = Y_test.numpy()
    
    if global_label_map is not None:
        mask = np.isin(Y_test, list(global_label_map.keys()))
        X_test = X_test[mask]
        Y_test = Y_test[mask]
        Y_test = np.array([global_label_map[l] for l in Y_test])
        
    if config.debug:
        limit = min(2000, len(X_test))
        indices = np.random.choice(len(X_test), limit, replace=False)
        X_test = X_test[indices]
        Y_test = Y_test[indices]
        
    return X_test, Y_test

def oh(y, num_classes):
    return np.eye(num_classes)[y.astype(int)]
