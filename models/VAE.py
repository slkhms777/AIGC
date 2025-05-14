import torch 
from torch import nn
from torch import optim

class VAE(nn.Module):
    def __init__(self, input_dim, num_hiddens, latent_dim):
        super().__init__()
        self.flt = nn.Flatten()
        # 编码器
        self.fc1 = nn.Linear(input_dim, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc_mu = nn.Linear(num_hiddens, latent_dim)
        self.fc_logvar = nn.Linear(num_hiddens, latent_dim)
        # 解码器
        self.fc3 = nn.Linear(latent_dim, num_hiddens)
        self.fc4 = nn.Linear(num_hiddens, num_hiddens)
        self.fc5 = nn.Linear(num_hiddens, input_dim)
        
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mu(x)
        log_var = self.fc_logvar(x)
        return mu, log_var

    # 重参数化
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var) # 标准差需要方差开根号
        eps = torch.randn_like(std)
        return mu + std * eps # 其中 eps ~ N(0, 1)

    def decode(self, z):
        x = self.relu(self.fc3(z))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        return x
    
    def forward(self, img): 
        x = self.flt(img)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, log_var

def vae_loss_function(recon_x, x, mu, log_var):
    # 重建损失（MSE 或 BCE）
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

def train_vae(model, dataloader, num_epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for img, _ in dataloader:
            img = img.to(device)
            img_flat = img.view(img.size(0), -1)
            optimizer.zero_grad()
            recon_x, mu, log_var = model(img)
            loss = vae_loss_function(recon_x, img_flat, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

def test_vae(model, dataloader, device='cpu'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            img_flat = img.view(img.size(0), -1)
            recon_x, mu, log_var = model(img)
            loss = vae_loss_function(recon_x, img_flat, mu, log_var)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Test Loss: {avg_loss:.4f}")


