import argparse
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，这个后端适用于生成图像文件但不显示它们
from torch.utils.data import DataLoader
import torch
import random
import glob
import joblib
from Dataset import Dataset
from cDCGAN_model import Generator, Discriminator
from torch.autograd import Variable
import torch.autograd as autograd


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
parser.add_argument("--latent_dim", type=int, default=1000, help="dimensionality of the latent space")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
# 添加WGAN - GP相关参数
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--lambda_gp", type=float, default=10, help="gradient penalty coefficient")
opt = parser.parse_args()
print(opt)

encoder = joblib.load('../model/label_encoder_2.joblib')
label_allow_list = encoder.classes_
path_to_files = '../data/stft_30w_clean/*'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def set_dataloader():
    # 生成训练数据
    train_file_paths = []
    train_labels = []
    for parent_path in glob.glob(path_to_files):
        # 判断类别是否参与训练
        label = os.path.basename(parent_path)
        if label not in label_allow_list:
            continue
        # 加载训练样本
        h5_file_path = os.path.join(parent_path, 'train/*/*.h5')
        for file_path in glob.glob(h5_file_path):
            train_file_paths.append(file_path)
            train_labels.append(label)

    train_labels = encoder.transform(train_labels)
    print('train nums:', len(train_labels))
    # 创建训练和验证数据生成器，加载所有允许类别的数据
    train_set = Dataset(train_file_paths, train_labels)
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    return train_loader


# 移除原有的BCELoss
# adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator(opt)
discriminator = Discriminator(opt)

generator.to(device)
discriminator.to(device)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
# 设置日志保存
csv_file = open('../log/cDCGAN.csv', 'w', newline='')
fieldnames = ['epoch', 'g_loss', 'd_loss', 'gradient_penalty', 'epoch_time', 'real_validity', 'fake_validity']
writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer_csv.writeheader()


# 定义梯度惩罚函数
def compute_gradient_penalty(D, real_samples, fake_samples, real_labels, fake_labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, real_labels)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# Training loop
train_loader = set_dataloader()
for epoch in range(opt.n_epochs):
    start_time = time.time()
    total_d_loss = 0
    total_g_loss = 0
    total_d_nums = 0
    total_g_nums = 0
    total_fake_validity = 0
    total_real_validity = 0
    total_gradient_penalty = 0
    for i, (imgs, labels) in enumerate(train_loader):
        batch_size = imgs.shape[0]
        generator.train()
        discriminator.train()

        # Configure input
        real_imgs = imgs.to(device)
        real_labels = labels.to(device)

        # Train Discriminator
        optimizer_D.zero_grad()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, opt.latent_dim, device=device)
        gen_labels = real_labels

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # 计算Wasserstein损失
        real_validity = discriminator(real_imgs, real_labels)
        fake_validity = discriminator(gen_imgs.detach(), gen_labels)
        total_fake_validity += torch.mean(fake_validity).item() * batch_size
        total_real_validity += torch.mean(real_validity).item() * batch_size
        # 计算梯度惩罚
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, real_labels,
                                                    gen_labels)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty
        total_gradient_penalty += gradient_penalty.item() * batch_size
        d_loss.backward()
        optimizer_D.step()

        total_d_loss += d_loss.item() * batch_size
        total_d_nums += batch_size

        if i % opt.n_critic == 0:
            # train Generator
            optimizer_G.zero_grad()
            # z = torch.randn(batch_size, opt.latent_dim, device=device)
            # gen_labels = torch.randint(0, opt.n_classes, (batch_size,), dtype=torch.long, device=device)
            gen_imgs = generator(z, gen_labels)
            fake_validity = discriminator(gen_imgs, gen_labels)
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            optimizer_G.step()
            total_g_loss += g_loss.item() * batch_size
            total_g_nums += batch_size

    end_time = time.time()
    epoch_time = end_time - start_time
    epoch_d_loss = total_d_loss / total_d_nums
    epoch_g_loss = total_g_loss / total_g_nums
    epoch_gradient_penalty = total_gradient_penalty / total_d_nums
    epoch_fake_validity = total_fake_validity / total_d_nums
    epoch_real_validity = total_real_validity / total_d_nums
    print(f"Epoch {epoch}, G Loss: {epoch_g_loss:.6f}, D Loss: {epoch_d_loss:.6f}, epoch_time: {epoch_time}, "
          f"gradient_penalty:{epoch_gradient_penalty:.6f},real_validity:{epoch_real_validity:.6f},"
          f"fake_validity:{epoch_fake_validity:.6f}")

    try:
        writer_csv.writerow(
            {'epoch': epoch, 'g_loss': epoch_g_loss, 'd_loss': epoch_d_loss, 'gradient_penalty': epoch_gradient_penalty,
             'epoch_time': epoch_time, 'fake_validity': epoch_fake_validity, 'real_validity': epoch_real_validity})
        csv_file.flush()  # 刷新文件缓冲区，确保数据立即写入文件
    except Exception as e:
        print(f"Error writing to CSV: {e}")

    # 保存模型
    torch.save(generator.state_dict(), f'../model/generator_model2.pth')
    torch.save(discriminator.state_dict(), f'../model/discriminator_model2.pth')

