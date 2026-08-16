import torch.nn as nn
import torch


class Generator(nn.Module):
    def _build_pixelshuffle_block(self, in_ch, out_ch,kernel_size=(3,3),padding=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 4, kernel_size=(3, 3), padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.2)
        )

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, 64)  # 将标签嵌入为和噪声维度一致的向量
        self.init_size = 49
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim + 64, 64 * self.init_size ** 2))

        self.layer1 = self._build_pixelshuffle_block(64, 64,kernel_size=(2,2),padding=0)
        self.layer2 = self._build_pixelshuffle_block(64, 32)
        self.layer3 = self._build_pixelshuffle_block(32, 16)
        self.layer4 = self._build_pixelshuffle_block(16, 8)

        self.conv1 = nn.Conv2d(8, opt.channels, kernel_size=(2, 2), stride=(1, 1), padding=0)
        self.Tanh = nn.Tanh()

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)  # 拼接噪声向量和标签嵌入向量
        x = self.l1(x)
        x = x.view(x.shape[0], 64, self.init_size, self.init_size)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv1(x)
        x = self.Tanh(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, out_dim1, out_dim2, stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1)
        self.ln1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1)
        self.ln2 = nn.InstanceNorm2d(out_channels)
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride),
                nn.InstanceNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.ln2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, input_shape):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.ln1 = nn.InstanceNorm2d(16)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(16, 16, (2, 2), 113, 97)
        self.layer2 = self._make_layer(16, 32, (2, 2), 57, 49)
        self.layer3 = self._make_layer(32, 64, (2, 2), 29, 25)
        self.layer4 = self._make_layer(64, 128, (2, 2), 15, 13)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        # for m in self.modules():
        #     if isinstance(m, ResidualBlock):
        #         nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, in_channels, out_channels, stride, out_dim1, out_dim2):
        layers = [ResidualBlock(in_channels, out_channels, out_dim1, out_dim2, stride)]
        layers += [ResidualBlock(out_channels, out_channels, out_dim1, out_dim2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, 125)  # 嵌入层，可根据实际调整维度
        self.encoder = ResNet((1, 900, 775))
        self.adv_layer = nn.Sequential(nn.Linear(128, 1))

    def forward(self, img, labels):
        c = self.label_embedding(labels)
        c = c.unsqueeze(1).unsqueeze(3)  # 调整标签维度
        c = c.expand(c.shape[0], 1, 125, 775)  # 扩展标签维度，与图像拼接
        x = torch.cat([img, c], dim=2)  # 拼接图像和标签嵌入向量
        out = self.encoder(x)
        validity = self.adv_layer(out)

        return validity

