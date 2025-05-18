import torch
import torch.nn as nn
import torch.optim as optim

data_path = '../../data'

# 条件生成对抗网络（CGAN）模型结构

class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 128),
            nn.ReLU(True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # labels: one-hot 向量
        x = torch.cat([noise, labels], dim=1)
        img = self.model(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_dim, label_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim + label_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # labels: one-hot 向量
        x = torch.cat([img, labels], dim=1)
        validity = self.model(x)
        return validity

# =================== CGAN 训练函数 ===================
def train_cgan(generator, discriminator, dataloader, noise_dim, label_dim, num_epochs=10, device='cpu'):
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for real_imgs, real_labels in dataloader:
            real_imgs = real_imgs.view(real_imgs.size(0), -1).to(device)
            batch_size = real_imgs.size(0)
            real_labels = real_labels.to(device)
            # one-hot 编码
            real_labels_onehot = torch.zeros(batch_size, label_dim, device=device)
            real_labels_onehot.scatter_(1, real_labels.unsqueeze(1), 1)

            real_targets = torch.ones(batch_size, 1, device=device)
            fake_targets = torch.zeros(batch_size, 1, device=device)

            # 训练判别器
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_labels = torch.randint(0, label_dim, (batch_size,), device=device)
            fake_labels_onehot = torch.zeros(batch_size, label_dim, device=device)
            fake_labels_onehot.scatter_(1, fake_labels.unsqueeze(1), 1)
            fake_imgs = generator(noise, fake_labels_onehot)

            outputs_real = discriminator(real_imgs, real_labels_onehot)
            outputs_fake = discriminator(fake_imgs.detach(), fake_labels_onehot)
            loss_d_real = criterion(outputs_real, real_targets)
            loss_d_fake = criterion(outputs_fake, fake_targets)
            loss_d = loss_d_real + loss_d_fake

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器
            noise = torch.randn(batch_size, noise_dim, device=device)
            gen_labels = torch.randint(0, label_dim, (batch_size,), device=device)
            gen_labels_onehot = torch.zeros(batch_size, label_dim, device=device)
            gen_labels_onehot.scatter_(1, gen_labels.unsqueeze(1), 1)
            gen_imgs = generator(noise, gen_labels_onehot)
            outputs = discriminator(gen_imgs, gen_labels_onehot)
            loss_g = criterion(outputs, real_targets)

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

# =============== CGAN 生成图片函数 ===============
def generate_cgan_samples(generator, noise_dim, label_dim, labels, device='cpu'):
    """
    labels: LongTensor, shape=[num_samples], 指定每个样本的类别
    """
    generator.to(device)
    generator.eval()
    num_samples = labels.size(0)
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim, device=device)
        labels_onehot = torch.zeros(num_samples, label_dim, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        fake_imgs = generator(noise, labels_onehot)
    return fake_imgs.cpu()

