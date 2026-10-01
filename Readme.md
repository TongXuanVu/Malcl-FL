## MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification

---

This repository contains code of the paper __MalCL: Leveraging GAN-Based Generative Replay to Combat Catastrophic Forgetting in Malware Classification__


### Dataset
The dataset for the experiments can be downloaded from [here](https://drive.google.com/drive/folders/1YGmxcQGqu22ZQuZccpD81WUBKHh7c3Jq?usp=sharing) and [Zenodo Repository](https://zenodo.org/records/14537891)

---

## 🚀 Chạy lại trên Kaggle (CIC-IoT23 100-client, full) — vá lỗ f1-weighted task 0-2

Bối cảnh: cac phien chay that dau tien (global round 1-81 = task 0 tron + task 1
tron + task 2 round 1-21) dung ban code CU chi log `acc`, `f1_mac`, `loss` moi
round. Code HIEN TAI trong `function.py`/`main.py` da tinh va ghi DU 14 cot
(gom `f1_wei`) cho MOI round, KHONG CAN sua gi them — chi can chay lai dung
doan 81 round nay.

### 1. Setup Notebook
* Enable **GPU T4 x2** hoac **GPU P100**.
* Add dataset: `tongxuanvu/iot100client`.

### 2. Clone & chay
```bash
!git clone https://github.com/TongXuanVu/Malcl-FL.git
%cd Malcl-FL/MalCL_torch

!python main.py \
    --train_data /kaggle/input/datasets/tongxuanvu/iot100client/100client \
    --test_data /kaggle/input/datasets/tongxuanvu/iot100client/100client/global_test_data.pt \
    --num_clients 100
```
(Cac tham so khac — `nb_task 6`, `init_classes 6`, `n_inc 6`, `final_classes 34`,
`num_rounds 30`, `seed_ 20`, `sample_select L1_C_Mean`, `Generator_loss FML` —
deu la mac dinh, KHONG can truyen lai, de dung y het lan chay goc.)

### 3. Dung lai sau round 81 (khong bat buoc chay het 180 round)
Checkpoint (`checkpoints/ckpt_task{T:02d}_latest.pth`) va CSV
(`metrics_round_by_round_live.csv`) duoc ghi/flush sau **MOI round**, nen dung
giua chung an toan, khong mat du lieu. Chi can **gui output duoi dong console**
dung luc dong `[Task 2 | Round 21/30 | Global 81] ...` xuat hien la co the bam
Stop — luc do da du 81 round can (task 0 + task 1 + task 2 round 1-21).

Neu khong muon canh console, cu de chay het 6 task (180 round) cung duoc, chi
la ton them GPU quota (~2-3 lan so voi dung o round 81).

### 4. Lay ket qua
```bash
!zip -r malcl_fl_task0-2_fix.zip MalCL_torch/logs/malcl_fl/cic_iot23/*/
```
Tai ve `metrics_round_by_round_live.csv` trong zip, gui lai de gop vao
`Tong hop ket qua/iot100/aggregate.py` (thay cho
`metrics_round_by_round_recovered.csv` cu, von chi co 3 cot).

---

* EMBER 2018 dataset    
We use the 2018 EMBER dataset, known for its challenging classification tasks, focusing on a subset of 337,035 malicious Windows PE files labeled by the top 100
malware families, each with over 400 samples. Features include file size, PE and COFF header details, DLL characteristics, imported and exported functions, and properties
like size and entropy, all computed using the feature hashing trick.
* AZ-Class    
The AZ-Class dataset contains 285,582 samples from 100 Android malware families, each with at least 200 samples. We extracted Drebin features (Arp et al.2014) from the apps, covering eight categories like hardware access, permissions, API calls, and network addresses.

### Environment
---
* pytorch version 2.0.1
* conda version 4.7.12
* python version 3.8.13
* NVIDIA RTX A6000
* CUDA version 11.4


## Experiment
1. command lines are follows:

CUDA_VISIBLE_DEVICES=0 python main.py

---


2. You can change the Hyper Parameters and options by command line:

You can see the changable options on MalCL_torch/arguments.py


if you want to set the sample selection to L1_B_Mean, the command line is

CUDA_VISIBLE_DEVICES=0 python main.py --sample_select L2_B_Mean



### Pipeline
---
![pipeline](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/pipeline_new.png)


### Architecture
---
* Generator


![Generator](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/Generator.png)


* Discriminator

  
![Discriminator](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/Discriminator.png)


* Classifier

  
![Classifier](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/Classifier.png)


### Results
---

![Table](https://github.com/MalwareReplayGAN/MalCL/blob/master/Repo_img/table.png)    
* Comparisons to Baseline and Prior Replay Models Using Ember Dataset. We report the mean accuracy scores (Mean) and minimum (Min) computed from every 11 tasks.    
 



