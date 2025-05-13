import torch 
from torch import nn
from torch import optim

class Encoder(nn.Module):
    def __init__(self, img_size, num_hiddens, in_channels=1):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(img_size * img_size * in_channels, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, num_hiddens)
        self.relu = nn.LeakyReLU()

    def forward(self, image):
        x = self.flt(image)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        latent_vec = self.fc3(x)
        return latent_vec

class Decoder(nn.Module):
    def __init__(self, img_size, num_hiddens, out_channels=1):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.fc1 = nn.Linear(num_hiddens, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc3 = nn.Linear(num_hiddens, img_size * img_size * out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, latent_vec):
        x = self.relu(self.fc1(latent_vec))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape(-1, self.out_channels, self.img_size, self.img_size)
        return x

class DAE(nn.Module):
    def __init__(self, img_size, num_hiddens, in_channels=1):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, num_hiddens=num_hiddens, in_channels=in_channels)
        self.decoder = Decoder(img_size=img_size, num_hiddens=num_hiddens, out_channels=in_channels)

    @staticmethod
    def add_noise(img, noise_factor=0.3):
        noisy_img = img + noise_factor * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        return noisy_img

    def fit(self, dataloader, num_epochs, device='cpu'):
        self.to(device)
        self.train()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.002)

        for epoch in range(num_epochs):
            for img, _ in dataloader:
                img = img.to(device)
                noisy_img = self.add_noise(img)
                optimizer.zero_grad()
                latent = self.encoder(noisy_img)
                output_img = self.decoder(latent)
                loss = criterion(img, output_img)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss : {loss.item():.4f}")

    def reconstruct(self, img, device='cpu'):
        self.eval()
        img = img.to(device)
        with torch.no_grad():
            latent_vec = self.encoder(img)
            output_img = self.decoder(latent_vec)
        return output_img.cpu()





