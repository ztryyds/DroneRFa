import torch
import torch.nn as nn
import torch.nn.functional as F


def position_coding(x):
    num_token, num_dims = x.size(-2), x.size(-1)
    p = torch.zeros((1, num_token, num_dims))
    t = torch.arange(num_token, dtype=torch.float32).reshape(-1, 1) / \
        torch.pow(1e4, torch.arange(0, num_dims, 2, dtype=torch.float32) / num_dims)
    p[:, :, 0::2] = torch.sin(t)
    p[:, :, 1::2] = torch.cos(t)
    return p


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)
        out += shortcut
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, input_shape):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(16, 16, stride=(2, 2))
        self.layer2 = self._make_layer(16, 32, stride=(2, 2))
        self.layer3 = self._make_layer(32, 64, stride=(2, 2))
        self.layer4 = self._make_layer(64, 128, stride=(2, 2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, ResidualBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, in_channels, out_channels, stride):
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        layers += [ResidualBlock(out_channels, out_channels)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x


class TransResNet(nn.Module):
    def __init__(self, input_shape):
        super(TransResNet, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.resnet = ResNet(input_shape)
        self.encoding_to_sa1 = nn.Sequential(
            nn.Linear(input_shape[2], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.encoding_to_sa2 = nn.Sequential(
            nn.Linear(input_shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.SA_1 = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True, dim_feedforward=512)
        self.SA_1 = nn.TransformerEncoder(self.SA_1, num_layers=3)
        self.SA_2 = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True, dim_feedforward=512)
        self.SA_2 = nn.TransformerEncoder(self.SA_2, num_layers=3)

        self.sa1_to_semantic = nn.Sequential(
            nn.Linear(input_shape[1] * 128, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.sa2_to_semantic = nn.Sequential(
            nn.Linear(input_shape[2] * 128, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        self.total_semantic = nn.Sequential(
            nn.Linear(128 * 3, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        out1 = self.resnet(x)
        y = x[:, 0, :, :]
        z = y.permute(0, 2, 1)
        y = self.encoding_to_sa1(y)
        z = self.encoding_to_sa2(z)
        y = y + position_coding(y).to(self.device)
        z = z + position_coding(z).to(self.device)
        y = self.SA_1(y)
        z = self.SA_2(z)

        y = y.view(y.shape[0], -1)
        z = z.view(z.shape[0], -1)
        y = self.sa1_to_semantic(y)
        z = self.sa2_to_semantic(z)
        semantic = torch.cat([out1, y], dim=1)
        semantic = torch.cat([semantic, z], dim=1)
        feature = self.total_semantic(semantic)
        feat = self.head(feature)
        feat = F.normalize(feat, dim=1)
        return feat, feature


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, feature):
        feat = self.head(feature)
        feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return self.softmax(logits), logits


class Classifier_gan(nn.Module):
    def __init__(self, num_classes):
        super(Classifier_gan, self).__init__()
        self.fc = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, feature):
        feat = self.head(feature)
        feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return self.softmax(logits), logits
