import torch
import torch.nn as nn
from torch.autograd import Variable
from copy import deepcopy
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys
import csv
import gc
import datetime
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

from models import Generator, Discriminator, Classifier
from function import test, get_dataloader, get_iter_test_dataset
from dynaconf import Dynaconf
from arguments import _parse_args
from setting import configurate, torch_setting
from _data import get_client_data, get_global_test_data, get_global_label_map
from train import report_result, compute_mean_logits_optimized
from train import average_weights, average_logits

# Setup config & arguments
config = Dynaconf()
args = _parse_args()
configurate(args, config)
torch_setting(config)

num_clients = 10 # Hardcoded as per FL standard in this workspace

# ── Results & Logging Setup ──
now = datetime.datetime.now().strftime('%d-%m-%y_%H-%M')
results_dir = os.path.join('logs', 'malcl_fl', 'cic_iot23', now)
os.makedirs(results_dir, exist_ok=True)
print(f"Results will be saved to: {results_dir}")

log_file = os.path.join(results_dir, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('MalCL-FL')

ckpt_dir = os.path.join(results_dir, 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)

# ── Test Mode Logic ──
if hasattr(args, 'mode') and args.mode == 'test':
    import glob
    test_ckpt_root = (args.test_checkpoint_dir if args.test_checkpoint_dir else ckpt_dir)
    ckpt_files = sorted(glob.glob(os.path.join(test_ckpt_root, 'ckpt_round*.pth')))
    if not ckpt_files:
        logger.error(f'[TEST] Khong tim thay checkpoint trong: {test_ckpt_root}')
    else:
        logger.info(f'[TEST] Tim thay {len(ckpt_files)} checkpoint(s). Bat dau evaluation...')
        global_label_map = get_global_label_map()
        X_test, Y_test = get_global_test_data(config, global_label_map)
        
        global_scaler = StandardScaler()
        global_scaler.partial_fit(X_test)
        
        _test_results = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for _cp in ckpt_files:
            _state = torch.load(_cp, map_location=device, weights_only=False)
            _task   = _state['task']
            _round  = _state['round']
            _global = _state['global_round']
            _C = Classifier(init_classes=config.init_classes)
            
            if _task > 0:
                _C = _C.expand_output_layer(config.init_classes, config.n_inc, _task)
                
            current_out = _C.output_dim
            target_out = args.final_classes
            if current_out > target_out:
                _C.output_dim = target_out
                _C.fc1 = nn.Linear(_C.fc1.in_features, _C.output_dim).to(device)
                _C.fc1_bn1 = nn.BatchNorm1d(_C.output_dim).to(device)

            _C.to(device)
            _C.load_state_dict(_state['classifier_state_dict'])
            _C.eval()
            
            config.task = _task
            config.n_class = _state['n_class']

            X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, config.n_class)
            test_loader, _ = get_dataloader(X_test_t, Y_test_t, config.batchsize, config.n_class, global_scaler, train=False)
            
            with torch.no_grad():
                _m = test(config, _C, test_loader)
            
            cm = _m.pop('confusion_matrix')
            
            logger.info(
                f'[TEST] {os.path.basename(_cp)} | Task {_task} R {_round} | '
                f"Acc: {_m['accuracy']:.2f}% | F1-Mac: {_m['f1_macro']:.2f}% | FPR(Wei): {_m['fpr_weighted']:.2f}%"
            )
            _test_results.append({'checkpoint': os.path.basename(_cp), 'task': _task,
                                   'round': _round, 'global_round': _global, **_m})
            
            if _cp == ckpt_files[-1]:
                try:
                    import seaborn as sns
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
                    plt.xlabel('Predicted')
                    plt.ylabel('Actual')
                    plt.title(f'Confusion Matrix - {os.path.basename(_cp)}')
                    plt.savefig(os.path.join(results_dir, 'test_malcl_final_cm.png'), dpi=150)
                    plt.close()
                except ImportError:
                    logger.warning("Seaborn not installed, skipping confusion matrix plot.")
                
        _df_test = pd.DataFrame(_test_results)
        
        # Rename columns to user requested format
        _df_test = _df_test.rename(columns={
            'task': 'task_id',
            'round': 'round_in_task',
            'accuracy': 'acc',
            'precision_micro': 'prec_mic',
            'precision_macro': 'prec_mac',
            'precision_weighted': 'prec_wei',
            'recall_micro': 'rec_mic',
            'recall_macro': 'rec_mac',
            'recall_weighted': 'rec_wei',
            'f1_micro': 'f1_mic',
            'f1_macro': 'f1_mac',
            'f1_weighted': 'f1_wei'
        })
        
        # Select and reorder desired columns
        desired_cols = [
            'task_id', 'round_in_task', 'global_round', 'acc', 
            'prec_mic', 'prec_mac', 'prec_wei', 
            'rec_mic', 'rec_mac', 'rec_wei', 
            'f1_mic', 'f1_mac', 'f1_wei', 'loss'
        ]
        _df_test = _df_test[[c for c in desired_cols if c in _df_test.columns]]
        
        _test_csv = os.path.join(results_dir, 'test_results.csv')
        _df_test.to_csv(_test_csv, index=False)
        logger.info(f'[TEST] Ket qua luu tai: {_test_csv}')
        
        # Plotting
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        def save_test_plot(x_vals, y_vals, metric_name, color, marker):
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, f'{color}-{marker}', linewidth=2, markersize=4)
            plt.xlabel('Global Round / Checkpoint Index')
            plt.ylabel(f'{metric_name} (%)' if metric_name != 'Loss' else 'Loss')
            plt.title(f'[TEST - MalCL FL] {metric_name} over Checkpoints')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_name = metric_name.lower().replace("-", "_")
            plt.savefig(os.path.join(results_dir, f'test_malcl_{safe_name}.png'), dpi=150)
            plt.close()

        def save_combined_plot(x_vals, y_mic, y_mac, y_wei, category_name):
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_mic, 'b-o', label=f'Micro-{category_name}', linewidth=1.5, markersize=3)
            plt.plot(x_vals, y_mac, 'g-s', label=f'Macro-{category_name}', linewidth=1.5, markersize=3)
            plt.plot(x_vals, y_wei, 'r-^', label=f'Weighted-{category_name}', linewidth=1.5, markersize=3)
            plt.xlabel('Global Round / Checkpoint Index')
            plt.ylabel(f'{category_name} (%)')
            plt.title(f'[TEST - MalCL FL] {category_name} (Micro vs Macro vs Weighted)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            safe_name = category_name.lower().replace("-", "_")
            plt.savefig(os.path.join(results_dir, f'test_malcl_combined_{safe_name}.png'), dpi=150)
            plt.close()

        if _test_results:
            x_axis = [r['global_round'] for r in _test_results]
            save_test_plot(x_axis, [r['accuracy'] for r in _test_results], 'Accuracy', 'b', 'o')
            save_test_plot(x_axis, [r.get('loss', 0) for r in _test_results], 'Loss', 'k', 'X')
            save_combined_plot(x_axis, 
                [r.get('precision_micro', 0) for r in _test_results], 
                [r.get('precision_macro', 0) for r in _test_results], 
                [r.get('precision_weighted', 0) for r in _test_results], 'Precision')
            save_combined_plot(x_axis, 
                [r.get('recall_micro', 0) for r in _test_results], 
                [r.get('recall_macro', 0) for r in _test_results], 
                [r.get('recall_weighted', 0) for r in _test_results], 'Recall')
            save_combined_plot(x_axis, 
                [r.get('f1_micro', 0) for r in _test_results], 
                [r.get('f1_macro', 0) for r in _test_results], 
                [r.get('f1_weighted', 0) for r in _test_results], 'F1-Score')
            
            save_test_plot(x_axis, [r.get('fpr_weighted', 0) for r in _test_results], 'FPR-Weighted', 'r', 'v')
            save_test_plot(x_axis, [r.get('fpr_macro', 0) for r in _test_results], 'FPR-Macro', 'm', 'x')
            logger.info(f'[TEST] Da ve bieu do don va ket hop vao: {results_dir}')
    sys.exit(0)

# ── Load Global Test Data ──
global_label_map = get_global_label_map()
X_test, Y_test = get_global_test_data(config, global_label_map)
config.feats_length = X_test.shape[1]

# Need a global scaler fitted on something?
# MalCL standard approach uses partial_fit during train. In FL, we could instantiate a global scaler and fit on all test data just to get dimensions right, but actually standard scaling per client might be better.
# Let's use a global scaler and fit it initially on the test set to avoid NaNs, or let clients fit their own. The old code used a single scaler updated progressively.
# For FL, we can let each client use its own scaler OR share one. We'll use a local scaler per client per task to keep it simple and independent, or just copy the global one.
global_scaler = StandardScaler()
global_scaler.partial_fit(X_test) # Initialize

# ── Models & Optimizers (Global) ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.device = device
G_global = Generator().to(device)
D_global = Discriminator().to(device)
C_global = Classifier(init_classes=config.init_classes).to(device)

criterion = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss()

# Import dynamic training functions based on config
if config.sample_select == 'L2_One_Hot': from sample_selection import common_vars, L2_One_Hot as get_replay_with_label
elif config.sample_select == 'L1_B_Mean': from sample_selection import common_vars, L1_B_Mean as get_replay_with_label
elif config.sample_select == 'L1_C_Mean': from sample_selection import common_vars, L1_C_Mean as get_replay_with_label

if config.Generator_loss == 'FML': from train import run_batch_FML as run_batch
elif config.Generator_loss == 'BCE': from train import run_batch_BCE as run_batch

# ── Resume Logic ──
start_task = 0
start_round = 0
global_round = 0
logits_real_global = None
past_Generator = None
past_Classifier = None
results_all = []
history_rounds = []
loss_history = []

if hasattr(args, 'resume') and args.resume and os.path.isfile(args.resume):
    # Resume logic similar to old main.py
    logger.info(f"==> Resuming from checkpoint: {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    start_task = checkpoint['task']
    start_round = checkpoint['round']
    global_round = checkpoint['global_round']
    target_n_class = checkpoint['n_class']
    
    if target_n_class > C_global.output_dim:
        C_global = C_global.expand_output_layer(config.init_classes, config.n_inc, start_task, target_dim=target_n_class)
    C_global.to(device)
    C_global.load_state_dict(checkpoint['classifier_state_dict'])
    G_global.load_state_dict(checkpoint['generator_state_dict'])
    D_global.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if 'logits_real' in checkpoint:
        logits_real_global = checkpoint['logits_real']
    if 'past_classifier_state_dict' in checkpoint and start_task > 0:
        past_Classifier = Classifier(init_classes=config.init_classes)
        for t_idx in range(1, start_task):
            past_Classifier = past_Classifier.expand_output_layer(config.init_classes, config.n_inc, t_idx)
        past_Classifier.load_state_dict(checkpoint['past_classifier_state_dict'])
        past_Classifier.to(device)
        past_Classifier.eval()
    if 'past_generator_state_dict' in checkpoint:
        past_Generator = Generator()
        past_Generator.load_state_dict(checkpoint['past_generator_state_dict'])
        past_Generator.to(device)
        past_Generator.eval()
    config.n_class = checkpoint['n_class']
    
    if start_task > 0:
        config.past_n_class = past_Classifier.output_dim if past_Classifier else (config.init_classes + config.n_inc * (start_task - 1))
        
    if start_round >= config.num_rounds:
        config.past_n_class = config.n_class
        config.n_class += config.n_inc
        if config.n_class > config.final_classes: config.n_class = config.final_classes
        start_task += 1
        start_round = 0

# ── Federated Training Loop ──
for task in range(start_task, config.nb_task):
    if task > start_task:
        config.past_n_class = config.n_class
        config.n_class += config.n_inc
        if config.n_class > config.final_classes: config.n_class = config.final_classes
    elif task == 0 and start_task == 0:
        config.n_class = config.init_classes
        
    config.task = task
    logger.info(f"--- Starting Task {task} ---")
    
    # 1. Inspect client data for this task to find max class
    max_class_this_task = config.n_class
    client_data_cache = {}
    for c_id in range(num_clients):
        cx, cy = get_client_data(config, c_id, task + 1, global_label_map)
        if cx is not None and len(cy) > 0:
            c_max = int(np.max(cy)) + 1
            if c_max > max_class_this_task: max_class_this_task = c_max
            client_data_cache[c_id] = (cx, cy)
            
    if max_class_this_task > C_global.output_dim:
        logger.info(f"Expanding Global Classifier: {C_global.output_dim} -> {max_class_this_task}")
        C_global = C_global.expand_output_layer(config.init_classes, config.n_inc, task, target_dim=max_class_this_task)
        C_global.to(device)
        config.n_class = max_class_this_task
    elif max_class_this_task < config.n_class:
        config.n_class = C_global.output_dim
        
    # Global test loader
    # Notice: using global_scaler for test loader to have consistent evaluation
    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, config.n_class)
    test_loader, _ = get_dataloader(X_test_t, Y_test_t, config.batchsize, config.n_class, global_scaler, train=False)
    
    current_start_round = start_round if task == start_task else 0
    num_rounds = 2 if config.debug else config.num_rounds
    
    for r in range(current_start_round, num_rounds):
        global_round += 1
        logger.info(f"\n--- Task {task}, Round {r + 1}/{num_rounds} (Global {global_round}) ---")
        
        local_G_weights = []
        local_D_weights = []
        local_C_weights = []
        round_client_losses = []
        
        # Client Loop
        for c_id in range(num_clients):
            if c_id not in client_data_cache:
                continue
                
            cx, cy = client_data_cache[c_id]
            # Copy global models for local training
            C_local = deepcopy(C_global).to(device)
            G_local = deepcopy(G_global).to(device)
            D_local = deepcopy(D_global).to(device)
            
            C_local.train(); G_local.train(); D_local.train()
            
            G_optimizer = optim.Adam(G_local.parameters(), lr=config.lr)
            D_optimizer = optim.Adam(D_local.parameters(), lr=config.lr)
            C_optimizer = optim.SGD(C_local.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
            
            # Local DataLoader
            client_scaler = deepcopy(global_scaler)
            train_loader, _ = get_dataloader(cx, cy, config.batchsize, config.n_class, client_scaler, train=True)
            config.nb_batch = len(train_loader)
            
            client_loss = []
            
            for epoch in range(config.local_epochs):
                if config.task > 0:
                    past_Generator.eval()
                    past_Classifier.eval()
                    with torch.no_grad():
                        synthetic, pred_label, logits_gen = common_vars(config, past_Generator, past_Classifier)
                        replay_pool, re_label_pool = get_replay_with_label(config, synthetic, pred_label, logits_gen, logits_real_global)
                        replay_size = replay_pool.size(0)
                else:
                    replay_size = 0
                    
                epoch_loss = []
                for n, (inputs, labels) in enumerate(train_loader):
                    inputs = inputs.float().to(device)
                    labels = labels.float().to(device)
                    if config.task > 0:
                        inputs = torch.cat((inputs, replay_pool), 0)
                        labels = torch.cat((labels, re_label_pool), 0)
                    
                    loss = run_batch(config, G_local, D_local, C_local, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, inputs, labels, replay_size)
                    epoch_loss.append(loss)
                client_loss.append(np.mean(epoch_loss))
                
            logger.info(f"  Client {c_id} Loss: {np.mean(client_loss):.4f}")
            round_client_losses.append(np.mean(client_loss))
            
            local_G_weights.append(G_local.state_dict())
            local_D_weights.append(D_local.state_dict())
            local_C_weights.append(C_local.state_dict())
            
        # FedAvg Aggregation
        if len(local_C_weights) > 0:
            C_global.load_state_dict(average_weights(local_C_weights))
            G_global.load_state_dict(average_weights(local_G_weights))
            D_global.load_state_dict(average_weights(local_D_weights))
            
        avg_round_loss = np.mean(round_client_losses) if round_client_losses else 0.0
        
        # End of Round Evaluation
        with torch.no_grad():
            metrics = test(config, C_global, test_loader)
            metrics.update({
                'task': task,
                'round': r + 1,
                'global_round': global_round,
                'loss': avg_round_loss
            })
            history_rounds.append(metrics)
            loss_history.append(avg_round_loss)
            logger.info(
                f"[Task {task} | Round {r+1}/{num_rounds} | Global {global_round}] "
                f"Acc: {metrics['accuracy']:.2f}% | F1-Mac: {metrics['f1_macro']:.2f}% | Loss: {avg_round_loss:.4f}"
            )
            
            _ckpt_name = f'ckpt_round{global_round:04d}_task{task:02d}_r{r+1:03d}_acc{metrics["accuracy"]:.1f}.pth'
            _ckpt_data = {
                'task': task, 'round': r + 1, 'global_round': global_round, 'n_class': config.n_class,
                'classifier_state_dict': C_global.state_dict(),
                'generator_state_dict': G_global.state_dict(),
                'discriminator_state_dict': D_global.state_dict(),
                'metrics': metrics,
            }
            if logits_real_global is not None: _ckpt_data['logits_real'] = logits_real_global
            if past_Classifier is not None: _ckpt_data['past_classifier_state_dict'] = past_Classifier.state_dict()
            if past_Generator is not None: _ckpt_data['past_generator_state_dict'] = past_Generator.state_dict()
            torch.save(_ckpt_data, os.path.join(ckpt_dir, _ckpt_name))
            
    # Post-Task: Compute logits_real for replay
    logger.info(f"Task {task} finished. Computing statistics for replay across clients...")
    client_logits = []
    for c_id in range(num_clients):
        if c_id not in client_data_cache: continue
        cx, cy = client_data_cache[c_id]
        c_scaler = deepcopy(global_scaler)
        c_loader, _ = get_dataloader(cx, cy, config.batchsize, config.n_class, c_scaler, train=True)
        c_logits = compute_mean_logits_optimized(config, C_global, c_loader)
        client_logits.append(c_logits)
        
    logits_real_global = average_logits(client_logits)
    
    past_Generator = deepcopy(G_global)
    past_Classifier = deepcopy(C_global)
    results_all.append(history_rounds[-1])
    logger.info(f"Task {task} fully completed.")
    
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# Reporting & Plotting
report_result(config, results_all, results_dir)

csv_path = os.path.join(results_dir, 'metrics.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f_csv:
    writer = csv.writer(f_csv)
    writer.writerow([
        'task', 'method', 'acc', 'prec_mic', 'prec_mac', 'prec_wei',
        'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei', 'fpr_wei', 'loss', 'avg_acc',
    ])
    for i, res in enumerate(results_all):
        avg_acc = sum(r['accuracy'] for r in results_all[:i+1]) / (i + 1)
        writer.writerow([
            res['task'], 'MalCL-FL', round(res['accuracy'], 4), round(res.get('precision_micro', 0), 4),
            round(res.get('precision_macro', 0), 4), round(res.get('precision_weighted', 0), 4),
            round(res.get('recall_micro', 0), 4), round(res.get('recall_macro', 0), 4),
            round(res.get('recall_weighted', 0), 4), round(res.get('f1_micro', 0), 4),
            round(res.get('f1_macro', 0), 4), round(res.get('f1_weighted', 0), 4), 
            round(res.get('fpr_weighted', 0), 4), round(res.get('loss', 0), 6), round(avg_acc, 4)
        ])

df_history = pd.DataFrame(history_rounds)
round_csv_path = os.path.join(results_dir, 'metrics_round_by_round.csv')
df_history.to_csv(round_csv_path, index=False)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(results_all)), [r['accuracy'] for r in results_all], marker='o', label='Accuracy')
plt.plot(range(len(results_all)), [r['f1_macro'] for r in results_all], marker='s', label='F1-Macro')
plt.xlabel('Task')
plt.ylabel('Percentage')
plt.title('Performance across Tasks')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(len(loss_history)), loss_history, color='red')
plt.xlabel('Round (Accumulated)')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.tight_layout()
plot_path = os.path.join(results_dir, 'training_report.png')
plt.savefig(plot_path)
plt.close()