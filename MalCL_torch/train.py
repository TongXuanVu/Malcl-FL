import torch
from function import get_iter_train_dataset, get_iter_test_dataset, get_dataloader
from torch.autograd import Variable
import os



################################################################################################

def vars_batch_train(config, x_, y_):
      x_ = x_.view([-1, config.feats_length])
      # y_real and y_fake mean fake or true in Discriminator
      y_real_ = Variable(torch.ones(x_.size(0), 1))
      y_fake_ = Variable(torch.zeros(x_.size(0), 1))
      if config.use_cuda:
        y_real_, y_fake_ = y_real_.to(config.device), y_fake_.to(config.device)
      z_ = torch.rand((x_.size(0), config.z_dim))
      x_, z_ = Variable(x_), Variable(z_)
      if config.use_cuda:
        x_, z_, y_ = x_.to(config.device), z_.to(config.device), y_.to(config.device)
      return x_, y_, z_, y_real_, y_fake_

################################################################################################

def update_Discriminator(D, G, D_optimizer, BCELoss, x_, z_, y_real_, y_fake_):
      D_optimizer.zero_grad()
      D_real, _ = D(x_)
      target_real = y_real_[:x_.size(0)]
      if D_real.shape != target_real.shape:
          print(f"\nDiscriminator Shape mismatch (Real): D_real={D_real.shape}, target={target_real.shape}, x_={x_.shape}")
      D_real_loss = BCELoss(D_real, target_real)
      G_ = G(z_)
      D_fake, _ = D(G_)
      target_fake = y_fake_[:x_.size(0)]
      if D_fake.shape != target_fake.shape:
          print(f"\nDiscriminator Shape mismatch (Fake): D_fake={D_fake.shape}, target={target_fake.shape}, G_={G_.shape}")
      D_fake_loss = BCELoss(D_fake, target_fake)
      D_loss = D_real_loss + D_fake_loss
      D_loss.backward()
      D_optimizer.step()



def update_Generator_BCE(D, G, G_optimizer, BCELoss, x_, z_, y_real_):
      G_optimizer.zero_grad()
      G_ = G(z_)
      D_fake, _ = D(G_)
      G_loss = BCELoss(D_fake, y_real_[:x_.size(0)])
      G_loss.backward()
      G_optimizer.step()



def update_Generator_FML(D, G, G_optimizer, x_, z_):
      G_optimizer.zero_grad()
      fake_data = G(z_)
      _, features_fake = D(fake_data)
      _, features_real_unl = D(x_)
      feature_mean_real = torch.mean(features_real_unl, dim=0)
      feature_mean_fake = torch.mean(features_fake, dim=0)
      G_loss = torch.mean(torch.abs(feature_mean_real - feature_mean_fake))
      G_loss.backward()
      G_optimizer.step()



def update_Classifier(config, C_optimizer, C, criterion, x_, y_, replay_size=0):
      C_optimizer.zero_grad()
      output = C(x_)
      
      # Convert one-hot targets to class indices for CrossEntropyLoss compatibility
      if len(y_.shape) > 1 and y_.shape[1] > 1:
          target = torch.max(y_, 1)[1].long()
      else:
          target = y_.long()

      if replay_size > 0:
          real_size = x_.size(0) - replay_size
          # Separate loss for real and replay samples
          loss_real = criterion(output[:real_size], target[:real_size])
          loss_replay = criterion(output[real_size:], target[real_size:])
          C_loss = loss_real + config.replay_weight * loss_replay
      else:
          C_loss = criterion(output, target)
          
      C_loss.backward()
      C_optimizer.step()
      return C_loss.item()


#####################################################################################################


def run_batch_BCE(config, G, D, C, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, x_, y_, replay_size=0):
  x_, y_, z_, y_real_, y_fake_ = vars_batch_train(config, x_, y_)
  for _ in range(config.g_iter):
    update_Generator_BCE(D, G, G_optimizer, BCELoss, x_, z_, y_real_)
    z_ = Variable(torch.rand((x_.size(0), config.z_dim))).to(config.device)
    update_Discriminator(D, G, D_optimizer, BCELoss, x_, z_, y_real_, y_fake_)
  return update_Classifier(config, C_optimizer, C, criterion, x_, y_, replay_size)


def run_batch_FML(config, G, D, C, G_optimizer, D_optimizer, C_optimizer, criterion, BCELoss, x_, y_, replay_size=0):
  x_, y_, z_, y_real_, y_fake_ = vars_batch_train(config, x_, y_)
  for _ in range(config.g_iter):
    update_Generator_FML(D, G, G_optimizer, x_, z_)
    z_ = Variable(torch.rand((x_.size(0), config.z_dim))).to(config.device)
    update_Discriminator(D, G, D_optimizer, BCELoss, x_, z_, y_real_, y_fake_)
  return update_Classifier(config, C_optimizer, C, criterion, x_, y_, replay_size)


################################################################################################

def col_arr(config, X_train_t):
  logits_collect = []
  if config.sample_select == 'L1_C_Mean':
    logits_collect = [[] for k in range(config.n_class)]
  return logits_collect 



def collect_logits(config, C, logits_collect, inputs, labels, batch):
  with torch.no_grad():
    C.eval()
    if config.sample_select == 'L1_B_Mean': logits_collect.append(C.get_logits(inputs).to("cpu"))
    elif config.sample_select == 'L1_C_Mean':
      temp_vec = C.get_logits(inputs).to("cpu")
      for ind, (inp, lab) in enumerate(zip(inputs, labels)):
        logits_collect[int(torch.max(lab, dim = 0)[1])].append(temp_vec[int(ind)])
  return logits_collect


####################################################################################################

def data_task(config, X_train, Y_train, X_test, Y_test, scaler):
    X_train_t, Y_train_t = get_iter_train_dataset(X_train,  Y_train, n_class=config.n_class, n_inc=config.n_inc, task=config.task)
    train_loader, scaler = get_dataloader(X_train_t, Y_train_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler)
    X_test_t, Y_test_t = get_iter_test_dataset(X_test, Y_test, n_class=config.n_class)
    test_loader, _ = get_dataloader(X_test_t, Y_test_t, batchsize=config.batchsize, n_class=config.n_class, scaler = scaler, train=False)
    return X_train_t, Y_train_t, train_loader, X_test_t, Y_test_t, test_loader, scaler

def mean_logits(config, logits_collect):
    if config.sample_select == 'L1_B_Mean':
      logits_real = []
      for i, row in enumerate(logits_collect):
        if row == []: 
           print(i)
           continue
        logits_real.append(torch.mean(row, dim = 0).float())
      logits_real = torch.stack(logits_real)
    elif config.sample_select == 'L1_C_Mean':
      logits_real = []
      for i, row in enumerate(logits_collect):
        if row == []: 
           print(i)
           continue
        logits_real.append(torch.mean(torch.stack(row).float(), dim=0))
      logits_real = torch.stack(logits_real)
    else: logits_real = None
    
    return logits_real

def compute_mean_logits_optimized(config, model, dataloader):
    """
    Tính toán logits trung bình một cách tối ưu về bộ nhớ (không lưu trữ tất cả logits).
    """
    model.eval()
    device = config.device
    
    # Lấy kích thước đầu ra thực tế của mô hình
    with torch.no_grad():
        inputs, _ = next(iter(dataloader))
        temp_logits = model.get_logits(inputs.to(device))
        logit_dim = temp_logits.shape[1]
    
    # Số lượng lớp cần tính (dựa trên số lớp hiện tại đã học)
    num_classes = config.n_class 
    
    sum_logits = torch.zeros((num_classes, logit_dim)).to(device)
    counts = torch.zeros(num_classes).to(device)
    
    print(f"Computing mean logits for {num_classes} classes (logit_dim={logit_dim})...")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            # labels là one-hot, lấy index lớp
            _, label_indices = torch.max(labels, dim=1)
            
            logits = model.get_logits(inputs)
            
            # Cộng dồn logits theo từng lớp
            for c in range(num_classes):
                mask = (label_indices == c)
                if mask.any():
                    sum_logits[c] += logits[mask].sum(dim=0)
                    counts[c] += mask.sum()
            
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i} batches...")
                
    # Tính trung bình
    logits_real_list = []
    for c in range(num_classes):
        if counts[c] > 0:
            logits_real_list.append(sum_logits[c] / counts[c])
        else:
            logits_real_list.append(torch.zeros(logit_dim).to(device))
            
    return torch.stack(logits_real_list)

def report_result(config, results_list, log_path=None):
    if log_path is None:
        log_path = os.getcwd()
    
    print("\n" + "="*50)
    print(f"Final Report - Generator Loss: {config.Generator_loss}, Sample: {config.sample_select}")
    print("="*50)
    
    acc_list = [r['accuracy'] for r in results_list]
    
    prec_micro_list = [r['precision_micro'] for r in results_list]
    prec_macro_list = [r['precision_macro'] for r in results_list]
    prec_weight_list = [r['precision_weighted'] for r in results_list]
    
    rec_micro_list = [r['recall_micro'] for r in results_list]
    rec_macro_list = [r['recall_macro'] for r in results_list]
    rec_weight_list = [r['recall_weighted'] for r in results_list]
    
    f1_micro_list = [r['f1_micro'] for r in results_list]
    f1_macro_list = [r['f1_macro'] for r in results_list]
    f1_weight_list = [r['f1_weighted'] for r in results_list]
    
    print(f"Accuracy for each task: {[round(a, 2) for a in acc_list]}")
    print(f"Global Average Acc: {sum(acc_list)/len(acc_list):.2f}%")
    
    file_name = f"results_{config.Generator_loss}_{config.sample_select}_seed{config.seed_}"
    with open(os.path.join(log_path, 'metrics_summary.txt'), 'w') as f:
        f.write("Task | Accuracy | Prec(Mac) | Rec(Mac) | F1(Mac) | F1(Wei)\n")
        f.write("-" * 65 + "\n")
        for i, res in enumerate(results_list):
            f.write(f"{i:4} | {res['accuracy']:8.2f} | {res['precision_macro']:9.2f} | {res['recall_macro']:8.2f} | {res['f1_macro']:7.2f} | {res['f1_weighted']:7.2f}\n")
        
        f.write("\nGlobal Averages:\n")
        f.write(f"Accuracy           : {sum(acc_list)/len(acc_list):.2f}%\n\n")
        
        f.write(f"Precision (Micro)  : {sum(prec_micro_list)/len(prec_micro_list):.2f}%\n")
        f.write(f"Precision (Macro)  : {sum(prec_macro_list)/len(prec_macro_list):.2f}%\n")
        f.write(f"Precision (Weight) : {sum(prec_weight_list)/len(prec_weight_list):.2f}%\n\n")
        
        f.write(f"Recall (Micro)     : {sum(rec_micro_list)/len(rec_micro_list):.2f}%\n")
        f.write(f"Recall (Macro)     : {sum(rec_macro_list)/len(rec_macro_list):.2f}%\n")
        f.write(f"Recall (Weight)    : {sum(rec_weight_list)/len(rec_weight_list):.2f}%\n\n")
        
        f.write(f"F1-Score (Micro)   : {sum(f1_micro_list)/len(f1_micro_list):.2f}%\n")
        f.write(f"F1-Score (Macro)   : {sum(f1_macro_list)/len(f1_macro_list):.2f}%\n")
        f.write(f"F1-Score (Weight)  : {sum(f1_weight_list)/len(f1_weight_list):.2f}%\n")

    print(f"Results saved to {log_path}")

import copy

def average_weights(w):
    """
    Returns the average of the weights (state_dicts).
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        if 'num_batches_tracked' in key:
            w_avg[key] = w_avg[key].true_divide(len(w))
        else:
            w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_logits(logits_list):
    """
    Averages a list of logits tensors across clients.
    """
    if not logits_list:
        return None
    
    avg_logits = torch.zeros_like(logits_list[0])
    valid_counts = torch.zeros(logits_list[0].shape[0]).to(logits_list[0].device)
    
    for client_logits in logits_list:
        if client_logits is None: continue
        for c in range(client_logits.shape[0]):
            if torch.sum(torch.abs(client_logits[c])) > 0:
                avg_logits[c] += client_logits[c]
                valid_counts[c] += 1
                
    for c in range(avg_logits.shape[0]):
        if valid_counts[c] > 0:
            avg_logits[c] /= valid_counts[c]
            
    return avg_logits

