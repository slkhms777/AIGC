import torch
from torch import nn
from torch import optim
import math

class Encoder(nn.Module):
    def __init__(self, img_size, num_hiddens, latent_dim):
        super().__init__()
        self.img_size = img_size
        self.num_hiddens = num_hiddens
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # 1 * 28 * 28
            nn.Conv2d(1, num_hiddens, kernel_size=4, stride=2, padding=1), # 14x14
            nn.LeakyReLU(),
            nn.Conv2d(num_hiddens, 2 * num_hiddens, kernel_size=4, stride=2, padding=1), # 7x7
            nn.LeakyReLU(),
            nn.Conv2d(2 * num_hiddens, latent_dim, kernel_size=3, stride=1, padding=1) # 7x7
        )
    def forward(self, x):
        z_e = self.net(x) # [batch_size, latent_dim, 7, 7]
        return z_e 


class VectorQuantization(nn.Module):
    def __init__(self, img_size, latent_dim, book_size):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.book_size = book_size
        self.codebook = nn.Embedding(book_size, latent_dim) # [k, dim]

    def forward(self, z_e):
        z_e = z_e.permute(0, 2, 3, 1).contiguous() # [batch_size, h, w, dim]
        flat_z = z_e.reshape(-1, self.latent_dim) # [batch_size*h*w, dim] = [N, dim]
        # (a-b)^2 = a^2 + b^2 - 2ab
        dist = ( 
            torch.sum(flat_z ** 2, dim=1, keepdim=True) # [N, 1]
            + torch.sum(self.codebook.weight ** 2, dim=1) # [k]    广播成[N, k]
            - 2 * flat_z @ self.codebook.weight.t()       # [N, k]
        )
        # dist[i][j] = [(ai1 - bj1)^2 + (ai2 - bj2)^2 ……]

        # 对于N中每个向量，找到L2距离最近的码本索引
        indices = torch.argmin(dist, dim=1) # [N]

        # 查表得到量化后的向量
        z_q = self.codebook(indices) # [N, dim]
        
        return z_q, indices


class Decoder(nn.Module):
    def __init__(self, batch_size, latent_dim, num_hiddens, img_channel=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_hiddens = num_hiddens
        self.img_channel = img_channel
        self.batch_size = batch_size
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_hiddens, kernel_size=4, stride=2, padding=1), # 7x7 -> 14x14
            nn.LeakyReLU(),
            nn.ConvTranspose2d(num_hiddens, 2 * num_hiddens, kernel_size=4, stride=2, padding=1), # 14x14 -> 28x28
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2 * num_hiddens, img_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z_q):
        feature_size = int(math.sqrt(z_q.shape[0] / self.batch_size))
        z = z_q.reshape(-1, feature_size, feature_size, self.latent_dim)
        z = z.permute(0, 3, 1, 2)
        x_recon = self.net(z)
        return x_recon

class VQVAE(nn.Module):
    def __init__(self, batch_size, img_size, img_channel, 
                 latent_dim, num_hiddens, book_size):
        super().__init__()
        self.encoder = Encoder(img_size=img_size, 
                               latent_dim=latent_dim, 
                               num_hiddens=num_hiddens)
        self.vq = VectorQuantization(img_size=img_size, latent_dim=latent_dim, book_size=book_size)
        self.decoder = Decoder(img_channel=img_channel, 
                               batch_size=batch_size,
                               num_hiddens=num_hiddens, 
                               latent_dim=latent_dim)
    def forward(self, img):
        z_e = self.encoder(img)
        print(f'z_e : {z_e.shape}')
        z_q, indices = self.vq(z_e)
        print(f'z_q ; {z_q.shape}')
        x_recon = self.decoder(z_q)
        print(f'x_recon : {x_recon.shape}')
        return x_recon
    

# ------------------ 损失函数 ------------------
# 1. 重构损失(Reconstruction Loss):
#    L_recon = ||x - x_recon||^2       # MSE或交叉熵，取决于数据类型
# 
# 2. 码本损失(Codebook Loss):
#    L_codebook = ||sg[z_e] - z_q||^2  # sg表示stop-gradient
# 
# 3. 承诺损失(Commitment Loss):
#    L_commit = β * ||z_e - sg[z_q]||^2 # β是权重系数
# 
# 4. 总损失:
#    L = L_recon + L_codebook + L_commit

def train_vqvae(model, train_loader, test_loader=None, num_epochs=10, 
                learning_rate=1e-3, beta=0.25, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    训练VQVAE模型
    
    参数:
        model: VQVAE模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器(可选)
        num_epochs: 训练轮数
        learning_rate: 学习率
        beta: 承诺损失权重系数
        device: 训练设备
    """
    print(f"Training on {device}")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练记录
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        recon_loss_sum = 0
        vq_loss_sum = 0
        commit_loss_sum = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            z_e = model.encoder(data)
            z_q, indices = model.vq(z_e)
            x_recon = model.decoder(z_q)
            
            # 计算损失
            recon_loss = torch.mean((data - x_recon) ** 2)
            vq_loss = torch.mean((z_e.detach() - z_q) ** 2)
            commit_loss = torch.mean((z_e - z_q.detach()) ** 2)
            
            loss = recon_loss + vq_loss + beta * commit_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新统计
            epoch_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            vq_loss_sum += vq_loss.item()
            commit_loss_sum += commit_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, Recon: {recon_loss.item():.4f}, "
                      f"VQ: {vq_loss.item():.4f}, Commit: {commit_loss.item():.4f}")
        
        # 记录平均训练损失
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = recon_loss_sum / len(train_loader)
        avg_vq = vq_loss_sum / len(train_loader)
        avg_commit = commit_loss_sum / len(train_loader)
        
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, "
              f"Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f}, Commit: {avg_commit:.4f}")
        
        # 评估测试集
        if test_loader is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(device)
                    z_e = model.encoder(data)
                    z_q, indices = model.vq(z_e)
                    x_recon = model.decoder(z_q)
                    
                    # 计算损失
                    recon_loss = torch.mean((data - x_recon) ** 2)
                    vq_loss = torch.mean((z_e - z_q) ** 2)
                    commit_loss = torch.mean((z_e - z_q) ** 2)
                    
                    loss = recon_loss + vq_loss + beta * commit_loss
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            print(f"Test Loss: {avg_test_loss:.4f}")
            
            # 可视化重构结果
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                visualize_reconstructions(model, test_loader, device, num_images=10)
    
    # 保存模型
    torch.save(model.state_dict(), f"vqvae_epoch{num_epochs}.pt")
    
    return train_losses, test_losses

# 可视化重构结果
def visualize_reconstructions(model, data_loader, device, num_images=10):
    """显示原始图像和重构图像"""
    import matplotlib.pyplot as plt
    
    model.eval()
    with torch.no_grad():
        # 获取一批数据
        data, _ = next(iter(data_loader))
        data = data[:num_images].to(device)
        
        # 重构
        z_e = model.encoder(data)
        z_q, indices = model.vq(z_e)
        reconstructions = model.decoder(z_q)
        
        # 转到CPU并准备显示
        data = data.cpu()
        reconstructions = reconstructions.cpu()
        
        # 创建图像网格
        fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))
        
        for i in range(num_images):
            # 原始图像
            axes[0, i].imshow(data[i][0], cmap='gray')
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # 重构图像
            axes[1, i].imshow(reconstructions[i][0], cmap='gray')
            axes[1, i].set_title('Reconstructed')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"reconstructions.png")
        plt.close()

# 示例：加载MNIST数据集并训练VQVAE
def train_vqvae_on_mnist(batch_size=128, num_epochs=20, latent_dim=16, 
                         hidden_dim=16, book_size=16):
    from torchvision import datasets, transforms
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # 加载MNIST
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VQVAE(batch_size=batch_size, 
                  img_size=28, 
                  img_channel=1, 
                  latent_dim=latent_dim, 
                  num_hiddens=hidden_dim, 
                  book_size=book_size)
    
    # 训练
    train_losses, test_losses = train_vqvae(model, train_loader, test_loader, 
                                           num_epochs=num_epochs, device=device)
    
    return model, train_losses, test_losses

# ==================== Prior 模型（PixelCNN） ====================
class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, padding=0, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.register_buffer('mask', torch.ones_like(self.conv.weight))
        
        # 创建掩码: mask_type='A'不包括当前位置, mask_type='B'包括当前位置
        h, w = kernel_size, kernel_size
        self.mask[:, :, h//2, w//2:] = 0  # 当前行右边的元素
        self.mask[:, :, h//2+1:, :] = 0   # 下面的所有行
        
        if mask_type == 'A':
            self.mask[:, :, h//2, w//2] = 0  # 当前位置 (仅A类掩码)
            
    def forward(self, x):
        self.conv.weight.data *= self.mask  # 确保每次前向传播使用掩码
        return self.conv(x)
        
        
class PixelCNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mask_type='B'):
        super().__init__()
        self.conv = MaskedConv2d(in_channels, out_channels*2, kernel_size, mask_type, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.gated_conv = nn.Conv2d(out_channels, out_channels, 1) # 1x1卷积
        
    def forward(self, x):
        out = self.conv(x)
        # 门控机制: tanh(x) * sigmoid(x)
        a, b = torch.chunk(out, 2, dim=1)
        out = torch.tanh(a) * torch.sigmoid(b)
        out = self.gated_conv(out)
        return out
        
        
class PixelCNN(nn.Module):
    def __init__(self, book_size,  latent_dim=64, n_layers=8):
        super().__init__()
        self.book_size = book_size 

        self.embedding = nn.Embedding(book_size, latent_dim)
        
        # 第一层使用mask_type='A'，不包括当前位置
        layers = [MaskedConv2d(latent_dim, latent_dim, 7, 'A', padding=3)]
        
        # 后续层使用mask_type='B'，包括当前位置
        for _ in range(n_layers):
            layers.append(PixelCNNLayer(latent_dim, latent_dim, kernel_size=7, mask_type='B'))
        
        self.layers = nn.Sequential(*layers)
        
        # 输出层: 预测每个位置的类别概率
        self.output_conv = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, 1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, book_size, 1)
        )
    
    def forward(self, x):
        x = self.embedding(x)  # [batch_size, h, w, latent_dim]
        x = x.permute(0, 3, 1, 2)  # [batch_size, latent_dim, h, w]
        
        x = self.layers(x)
        logits = self.output_conv(x)  # [batch_size, book_size, h, w]
        
        return logits

