import torch
from torch import nn
import torch.optim as optim

data_path = '../../data'

batch_size = 256
noise_size = 32
num_hiddens = 256
shape = (batch_size, noise_size)
mean = 0
std = 1

class Generator(nn.Module):
    def __init__(self, img_size, noise_size, num_hiddens, out_channels=1):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.fc1 = nn.Linear(noise_size, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, img_size * img_size * out_channels)
        self.relu = nn.LeakyReLU()
    
    def forward(self, noise):
        x = self.relu(self.fc1(noise))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class Discriminator(nn.Module):
    def __init__(self, img_size, num_hiddens, in_channels=1):
        super().__init__()
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size * in_channels, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.flt(image)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(x)
        return x

def train_gan(generator, discriminator, dataloader, noise_size, num_epochs=10, device='cpu'):
    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # 训练判别器
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_imgs = generator(noise)
            outputs_real = discriminator(real_imgs)
            outputs_fake = discriminator(fake_imgs.detach())
            loss_d_real = criterion(outputs_real, real_labels)
            loss_d_fake = criterion(outputs_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # 训练生成器
            noise = torch.randn(batch_size, noise_size, device=device)
            fake_imgs = generator(noise)
            outputs = discriminator(fake_imgs)
            loss_g = criterion(outputs, real_labels) # 希望判别器认为生成图片为真

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss D: {loss_d.item():.4f}, Loss G: {loss_g.item():.4f}")

def generate_samples(generator, noise_size, num_samples=16, device='cpu'):
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_size, device=device)
        fake_imgs = generator(noise)
    return fake_imgs.cpu()





