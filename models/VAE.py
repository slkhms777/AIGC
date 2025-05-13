import torch 
from torch import nn
from torch import optim

class VAE(nn.Module):
    def __init__(self, img_size, num_hiddens, latent_dim):
        super().__init__()
        self.flt = nn.Flatten()

    def encode(self, x):
        return x

    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps # 其中 eps ~ N(0, 1)

    def decoder(self, x):
        return x
    
    def forward(self, img): 
        x = self.flt(img)
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)



