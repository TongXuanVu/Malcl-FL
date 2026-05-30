import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
# Removed from data_ import oh
def oh(y, num_classes):
    if isinstance(y, torch.Tensor):
        y_long = y.clone().detach().long()
    else:
        y_long = torch.tensor(y, dtype=torch.long)
    return torch.nn.functional.one_hot(y_long, num_classes=num_classes).float()
import time
import copy


def class_pick_rand(config, Y_train, Y_test):

    torch.manual_seed(config.seed_)

    class_arr = np.arange(config.final_classes)
    indices = torch.randperm(config.final_classes)
    class_arr = torch.index_select(torch.Tensor(class_arr), dim=0, index=indices)

    class_arr = np.array(class_arr)
    class_arr = list(class_arr)

    Y_train_ = copy.deepcopy(Y_train)
    Y_test_ = copy.deepcopy(Y_test)

    for i in range(0, config.final_classes):
        Y_train[np.where(Y_train_ == class_arr[i])] = i
        Y_test[np.where(Y_test_ == class_arr[i])] = i

    print("class_pick_rand")
    print(class_arr)


    return Y_train, Y_test


def get_iter_train_dataset(x, y, n_class=None, n_inc=None, task=None):
   
   if task is not None:
    if task == 0:
       selected_indices = np.where(y < n_class)[0] 
    else:
       start = n_class - n_inc
       end = n_class
       selected_indices = np.where((y >= start) & (y < end))
    
    return x[selected_indices], y[selected_indices]


def get_iter_test_dataset(x, y, n_class):
    selected_indices = np.where(y < n_class)[0] 
    return x[selected_indices], y[selected_indices]

def get_dataloader(x, y, batchsize, n_class, scaler, train = True):

    y_ = np.array(y, dtype=int)

    if train: 
        unique_classes = np.unique(y_)
        class_sample_count = np.array([len(np.where(y_ == t)[0]) for t in unique_classes])
        weight = 1. / class_sample_count
        weight_map = {t: weight[i] for i, t in enumerate(unique_classes)}
        samples_weight = np.array([weight_map[t] for t in y_])
    
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    

    x_ = torch.from_numpy(x).type(torch.FloatTensor)
    y_ = torch.from_numpy(y_).type(torch.FloatTensor)

    # Scaling
    if train: scaler = scaler.partial_fit(x_)
    x_ = scaler.transform(x_)
    x_ = torch.FloatTensor(x_)
    
    # One-hot Encoding
    y_oh = oh(y_, num_classes=n_class)
    y_oh = torch.Tensor(y_oh)

    data_tensored = torch.utils.data.TensorDataset(x_, y_oh)
    if train: Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=0, sampler=sampler)
    else: Loader = torch.utils.data.DataLoader(data_tensored, batch_size=batchsize, num_workers=0)
    
    return Loader, scaler


def test(config, model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(config.device))
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            
            loss = F.cross_entropy(outputs, labels_max.to(config.device))
            total_loss += loss.item() * labels_max.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_max.cpu().numpy())
            
            total += labels_max.size(0)
            correct += (predicted.cpu() == labels_max.cpu()).sum().item()
            
    accuracy = (correct / total) * 100
    avg_loss = total_loss / total if total > 0 else 0.0
    
    precision_micro = precision_score(all_labels, all_preds, average='micro', zero_division=0) * 100
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    
    recall_micro = recall_score(all_labels, all_preds, average='micro', zero_division=0) * 100
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0) * 100
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
    
    # Confusion Matrix & FPR
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(config.n_class))
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    
    fpr_per_class = fp / (fp + tn + 1e-7)
    
    counts = np.bincount(all_labels, minlength=config.n_class)[:config.n_class]
    if np.sum(counts) > 0:
        fpr_weighted = np.average(fpr_per_class, weights=counts) * 100
    else:
        fpr_weighted = 0.0
        
    fpr_macro = np.mean(fpr_per_class) * 100

    print(f'Test Results - Acc: {accuracy:.2f}%, F1(Mac): {f1_macro:.2f}%, FPR(Wei): {fpr_weighted:.2f}%')

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'fpr_weighted': fpr_weighted,
        'fpr_macro': fpr_macro,
        'confusion_matrix': cm
    }
