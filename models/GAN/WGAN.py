import torch
import torch.nn as nn
import torch.optim as optim

# ========== WGAN模型结构 ==========
class Generator(nn.Module):
    def __init__(self, noise_dim, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
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

    def forward(self, noise):
        img = self.model(noise)
        return img

class Critic(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# ========== WGAN训练函数 ==========
def train_wgan(generator, critic, dataloader, noise_dim, num_epochs=10, device='cpu', n_critic=5, clip_value=0.01):
    generator.to(device)
    critic.to(device)
    generator.train()
    critic.train()
    optimizer_g = optim.RMSprop(generator.parameters(), lr=0.00005)
    optimizer_c = optim.RMSprop(critic.parameters(), lr=0.00005)

    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.view(real_imgs.size(0), -1).to(device)
            batch_size = real_imgs.size(0)

            # 训练Critic
            for _ in range(n_critic):
                noise = torch.randn(batch_size, noise_dim, device=device)
                fake_imgs = generator(noise).detach()
                loss_c = -(torch.mean(critic(real_imgs)) - torch.mean(critic(fake_imgs)))

                optimizer_c.zero_grad()
                loss_c.backward()
                optimizer_c.step()

                # 权重裁剪
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # 训练Generator
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_imgs = generator(noise)
            loss_g = -torch.mean(critic(fake_imgs))

            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss C: {loss_c.item():.4f}, Loss G: {loss_g.item():.4f}")

# ========== WGAN生成图片函数 ==========
def generate_wgan_samples(generator, noise_dim, num_samples=16, device='cpu'):
    generator.to(device)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim, device=device)
        fake_imgs = generator(noise)
    return fake_imgs.cpu()