import argparse
import os

import h5py
import joblib
import numpy as np
import torch
from torch.autograd import Variable
from cDCGAN_model import Generator
import random

def set_seed(seed):
    random.seed(seed)  # 设置Python随机库的种子
    torch.manual_seed(seed)  # 设置PyTorch的种子
    np.random.seed(seed)  # 设置NumPy的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 如果有CUDA，设置CUDA的种子
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，设置所有CUDA设备的种子
    torch.backends.cudnn.deterministic = True  # 确保cudnn的确定性
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的基准测试模式
    torch.backends.cudnn.enabled = False  # 禁用cudnn
    print('Random seed :', seed)

my_seed = 42
set_seed(my_seed)
parser = argparse.ArgumentParser()
# 新增条件类别数量参数，比如MNIST数据集有10类数字
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for conditional information")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
# parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
# parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# 添加WGAN - GP相关参数
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty coefficient")
opt = parser.parse_args()
print(opt)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = joblib.load('../model/label_encoder_1.joblib')
label_allow_list = encoder.classes_
count_per_label = 100
generator = Generator(opt).to(device)
model_weights = torch.load(f'../model/generator_model1.pth')
generator.load_state_dict(model_weights)
generator.eval()
batch_size = 50

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Create output directories
base_dir = '../data/gen_data'
for label in label_allow_list:
    label_base_dir = os.path.join(base_dir, label)
    os.makedirs(label_base_dir, exist_ok=True)
    encode_label = encoder.transform([label])[0]
    for i in range(count_per_label):
        noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        labels = torch.full((batch_size,), encode_label)
        gen_labels = Variable(labels.type(LongTensor))
        gen_imgs = generator(noise, gen_labels)
        for index,img in enumerate(gen_imgs):
            idx = i*batch_size + index
            stft_output_filename = os.path.join(label_base_dir, f'gen_{idx}_stft.h5')
            img = img.detach().cpu().numpy()
            with h5py.File(stft_output_filename, 'w') as stft_fw:
                stft_fw.create_dataset('STFT Magnitude', data=img.astype(np.float32))
            print(stft_output_filename,'has saved')

