#!/bin/bash
#SBATCH -J train_new # 作业名
#SBATCH -p dgx1 # 提交到默认的defq 队列
#SBATCH -N 1 # 结点数
#SBATCH --ntasks-per-node=1 # 每个节点的进程数
#SBATCH --cpus-per-task=6 # 每个进程占用cpu核心数
#SBATCH -t 48:00:00 # 任务最大运行时间
#SBATCH --gres=gpu:1  # gpu数量

python -u SupCon_Transformer_train.py
python -u Classifier_train.py
python -u Classifier_test.py

python -u Openmax_data.py
python -u Openmax_MAV.py

python -u gen_test.py
python -u move.py

python -u Classifier_train_gan.py
python -u Classifier_test_gan.py

python -u G_Openmax_data.py
python -u G_Openmax_MAV.py