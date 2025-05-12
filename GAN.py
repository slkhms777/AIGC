import torch
from torch import nn
import torch.optim as optim

batch_size = 256
noise_size = 32
num_hiddens = 256
shape = (batch_size, noise_size)
mean = 0
std = 1
# noise = torch.normal(mean=torch.full(shape, mean), std=torch.full(shape, std))
# noise = torch.randn(batch_size, noise_size)

# 以灰度图像为例，num of channel=1
class Generator(nn.Module):
    def __init__(self, img_size, noise_size, num_hiddens):
        super().__init__()
        self.fc1 = nn.Linear(noise_size, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, img_size * img_size)
        self.relu = nn.LeakyReLU()
    
    def forward(self, noise):
        x = self.relu(self.fc1(noise))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 1, self.img_size, self.img_size)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size, num_hiddens):
        super().__init__()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.flt(image)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(x)
        return x


class GAN(nn.Module):
    def __init__(self, img_size, noise_size, num_hiddens):
        super().__init__()
        self.generator = Generator(img_size=img_size, noise_size=noise_size, num_hiddens=num_hiddens)
        self.discriminator = Discriminator(img_size=img_size, num_hiddens=num_hiddens)
        self.img_size = img_size
        self.noise_size = noise_size

    def train(self, dataloader, num_epochs=10, device='cpu'):
        self.generator.to(device)
        self.discriminator.to(device)
        self.generator.train()
        self.discriminator.train()
        criterion = nn.BCELoss()
        optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002)
        optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002)

        for epoch in range(num_epochs):
            for real_imgs, _ in dataloader:
                real_imgs = real_imgs.to(device)
                batch_size = real_imgs.size(0)
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # 训练判别器
                noise = torch.randn(batch_size, self.noise_size, device=device)
                fake_imgs = self.generator(noise)
                outputs_real = self.discriminator(real_imgs)
                outputs_fake = self.discriminator(fake_imgs.detach())
                loss_d_real = criterion(outputs_real, real_labels)
                loss_d_fake = criterion(outputs_fake, fake_labels)
                loss_d = loss_d_real + loss_d_fake

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

                # 训练生成器
                noise = torch.randn(batch_size, self.noise_size, device=device)
                fake_imgs = self.generator(noise)
                outputs = self.discriminator(fake_imgs)
                loss_g = criterion(outputs, real_labels) # 希望判别器认为生成图片为真

                optimizer_g.zero_grad()
                loss_g.backward()
                optimizer_g.step()

            print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

    def generate(self, num_samples=16, device='cpu'):
        self.generator.to(device)
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.noise_size, device=device)
            fake_imgs = self.generator(noise)
        return fake_imgs.cpu()





