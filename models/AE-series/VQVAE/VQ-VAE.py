import torch
from torch import nn
from torch import optim
import math

# ======================= 工具函数 =======================
# region 检查点函数
def save_checkpoint(vqvae=None, pixelcnn=None, vqvae_optimizer=None, pixelcnn_optimizer=None, 
                   epoch=0, hyperparams=None, checkpoint_dir='./checkpoints'):
    """
    保存训练检查点
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 如果是DataParallel，取其module
    if vqvae is not None and isinstance(vqvae, nn.DataParallel):
        vqvae_state = vqvae.module.state_dict()
    elif vqvae is not None:
        vqvae_state = vqvae.state_dict()
    else:
        vqvae_state = None

    if pixelcnn is not None and isinstance(pixelcnn, nn.DataParallel):
        pixelcnn_state = pixelcnn.module.state_dict()
    elif pixelcnn is not None:
        pixelcnn_state = pixelcnn.state_dict()
    else:
        pixelcnn_state = None

    checkpoint = {
        'epoch': epoch,
        'hyperparams': hyperparams
    }
    if vqvae_state is not None:
        checkpoint['vqvae_state_dict'] = vqvae_state
    if vqvae_optimizer is not None:
        checkpoint['vqvae_optimizer'] = vqvae_optimizer.state_dict()
    if pixelcnn_state is not None:
        checkpoint['pixelcnn_state_dict'] = pixelcnn_state
    if pixelcnn_optimizer is not None:
        checkpoint['pixelcnn_optimizer'] = pixelcnn_optimizer.state_dict()
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存至 {checkpoint_path}")


def load_checkpoint(checkpoint_path, vqvae=None, pixelcnn=None, vqvae_optimizer=None, 
                   pixelcnn_optimizer=None, device='cuda'):
    """
    加载训练检查点
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if vqvae is not None and 'vqvae_state_dict' in checkpoint:
        # 如果模型是DataParallel，直接load；否则用module的参数
        if isinstance(vqvae, nn.DataParallel):
            vqvae.module.load_state_dict(checkpoint['vqvae_state_dict'])
        else:
            vqvae.load_state_dict(checkpoint['vqvae_state_dict'])
    
    if vqvae_optimizer is not None and 'vqvae_optimizer' in checkpoint:
        vqvae_optimizer.load_state_dict(checkpoint['vqvae_optimizer'])
    
    if pixelcnn is not None and 'pixelcnn_state_dict' in checkpoint:
        if isinstance(pixelcnn, nn.DataParallel):
            pixelcnn.module.load_state_dict(checkpoint['pixelcnn_state_dict'])
        else:
            pixelcnn.load_state_dict(checkpoint['pixelcnn_state_dict'])
    
    if pixelcnn_optimizer is not None and 'pixelcnn_optimizer' in checkpoint:
        pixelcnn_optimizer.load_state_dict(checkpoint['pixelcnn_optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    hyperparams = checkpoint.get('hyperparams', {})
    
    print(f"检查点已从 {checkpoint_path} 加载，轮次: {epoch}")
    return epoch, hyperparams
# endregion


# ======================= VQ-VAE 模型 =======================
# region VQ-VAE 模型定义
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
        # dist[i][j] = [(ai1 - bj1)^2 + (ai2 - bj2)^2 …….

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
        self.feature_size = 7  # 固定特征图尺寸
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_hiddens, kernel_size=4, stride=2, padding=1), # 7x7 -> 14x14
            nn.LeakyReLU(),
            nn.ConvTranspose2d(num_hiddens, 2 * num_hiddens, kernel_size=4, stride=2, padding=1), # 14x14 -> 28x28
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2 * num_hiddens, img_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, z_q):
        # 直接用固定的 feature_size
        z = z_q.view(-1, self.feature_size, self.feature_size, self.latent_dim)
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
        z_q, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon
# endregion


# ======================= PixelCNN Prior 模型 =======================
# region Prior 模型定义
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
    def __init__(self, book_size, latent_dim=64, n_layers=8):
        super().__init__()
        self.book_size = book_size 

        # 嵌入层: 将索引转换为特征向量
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
# endregion


# ======================= 训练函数 =======================
# region 训练函数
def train_vqvae(model, train_loader, test_loader=None, num_epochs=10, 
                learning_rate=0.0001, beta=0.5, device="cuda" if torch.cuda.is_available() else "cpu",
                checkpoint_dir='./checkpoints', checkpoint_freq=5, resume_from=None):
    """
    训练VQVAE模型，支持检查点保存和恢复
    
    参数:
        model: VQVAE模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器(可选)
        num_epochs: 训练轮数
        learning_rate: 学习率
        beta: 承诺损失权重系数
        device: 训练设备
        checkpoint_dir: 检查点保存目录
        checkpoint_freq: 检查点保存频率（轮次）
        resume_from: 检查点文件路径（可选），用于恢复训练
    """
    print(f"Training on {device}")
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张GPU对vqvae进行训练")
        model = nn.DataParallel(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    
    # 从检查点恢复训练
    if resume_from:
        start_epoch, _ = load_checkpoint(resume_from, vqvae=model, vqvae_optimizer=optimizer, device=device)
        start_epoch += 1  # 从下一个epoch开始
    
    # 训练记录
    train_losses = []
    test_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        recon_loss_sum = 0
        vq_loss_sum = 0
        commit_loss_sum = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # 前向传播 - 直接使用模型的forward方法而不是单独访问子模块
            x_recon = model(data)
            
            # 获取中间结果 - 需要考虑DataParallel包装情况
            if isinstance(model, nn.DataParallel):
                z_e = model.module.encoder(data)
                z_q, indices = model.module.vq(z_e)
            else:
                z_e = model.encoder(data)
                z_q, indices = model.vq(z_e)
            
            # 修正z_q形状
            batch, latent_dim, h, w = z_e.shape
            z_q = z_q.view(batch, h, w, latent_dim).permute(0, 3, 1, 2).contiguous()  # [batch, latent_dim, 7, 7]
            
            # 计算损失
            # 1. 重构损失(Reconstruction Loss)
            recon_loss = torch.mean((data - x_recon) ** 2)
            # 2. 码本损失(Codebook Loss) - sg[z_e]表示stop-gradient
            vq_loss = torch.mean((z_e.detach() - z_q) ** 2)
            # 3. 承诺损失(Commitment Loss) - sg[z_q]表示stop-gradient
            commit_loss = torch.mean((z_e - z_q.detach()) ** 2)
            
            # 4. 总损失
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
                      f"Loss: {loss.item():.8f}, Recon: {recon_loss.item():.8f}, "
                      f"VQ: {vq_loss.item():.8f}, Commit: {commit_loss.item():.8f}")
        
        # 记录平均训练损失
        avg_loss = epoch_loss / len(train_loader)
        avg_recon = recon_loss_sum / len(train_loader)
        avg_vq = vq_loss_sum / len(train_loader)
        avg_commit = commit_loss_sum / len(train_loader)
        
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.8f}, "
              f"Recon: {avg_recon:.8f}, VQ: {avg_vq:.8f}, Commit: {avg_commit:.8f}")
        
        # 评估测试集
        if test_loader is not None:
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for data, _ in test_loader:
                    data = data.to(device)
                    # 使用完整的forward流程
                    x_recon = model(data)
                    
                    # 获取中间结果 - 需要考虑DataParallel包装情况
                    if isinstance(model, nn.DataParallel):
                        z_e = model.module.encoder(data)
                        z_q, indices = model.module.vq(z_e)
                    else:
                        z_e = model.encoder(data)
                        z_q, indices = model.vq(z_e)
                    
                    # 修正z_q形状
                    batch, latent_dim, h, w = z_e.shape
                    z_q = z_q.view(batch, h, w, latent_dim).permute(0, 3, 1, 2).contiguous()  # [batch, latent_dim, 7, 7]
                    
                    # 计算损失
                    recon_loss = torch.mean((data - x_recon) ** 2)
                    vq_loss = torch.mean((z_e - z_q) ** 2)
                    commit_loss = torch.mean((z_e - z_q) ** 2)
                    
                    loss = recon_loss + vq_loss + beta * commit_loss
                    test_loss += loss.item()
            
            avg_test_loss = test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            print(f"Test Loss: {avg_test_loss:.8f}")
        
        # 定期保存检查点
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
            hyperparams = {
                'learning_rate': learning_rate,
                'beta': beta,
                'num_epochs': num_epochs
            }
            save_checkpoint(vqvae=model, vqvae_optimizer=optimizer, epoch=epoch+1, 
                           hyperparams=hyperparams, checkpoint_dir=checkpoint_dir)
    
    return train_losses, test_losses
def train_vqvae_prior(pixelcnn, vqvae, dataloader, num_epochs, device,
                     checkpoint_dir='./checkpoints', checkpoint_freq=5, resume_from=None):
    """
    训练PixelCNN Prior模型，支持检查点保存和恢复
    
    参数:
        pixelcnn: PixelCNN模型
        vqvae: 已训练好的VQ-VAE模型
        dataloader: 原始数据加载器
        num_epochs: 训练轮数
        device: 训练设备
        checkpoint_dir: 检查点保存目录
        checkpoint_freq: 检查点保存频率（轮次）
        resume_from: 检查点文件路径（可选），用于恢复训练
    """
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张GPU对prior进行训练")
        pixelcnn = nn.DataParallel(pixelcnn)
        vqvae = nn.DataParallel(vqvae)  # 推理也用DataParallel，保证保存/加载一致

    pixelcnn = pixelcnn.to(device)
    vqvae = vqvae.to(device)
    
    vqvae.eval()  # Prior训练时VQ-VAE不更新
    optimizer = optim.Adam(pixelcnn.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    
    # 从检查点恢复训练
    if resume_from:
        start_epoch, _ = load_checkpoint(resume_from, pixelcnn=pixelcnn, 
                                        pixelcnn_optimizer=optimizer, device=device)
        start_epoch += 1  # 从下一个epoch开始

    # 训练记录
    train_losses = []
    
    for epoch in range(start_epoch, num_epochs):
        pixelcnn.train()
        epoch_loss = 0
        for data in dataloader:
            indices = data[0].to(device)  # indices: [batch, h, w], long
            logits = pixelcnn(indices)    # logits: [batch, book_size, h, w]
            loss = criterion(logits, indices.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Prior Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.8f}")
        train_losses.append(epoch_loss / len(dataloader))
            
        # 定期保存检查点
        if (epoch + 1) % checkpoint_freq == 0 or (epoch + 1) == num_epochs:
            hyperparams = {
                'learning_rate': 1e-3,
                'num_epochs': num_epochs
            }
            save_checkpoint(pixelcnn=pixelcnn, pixelcnn_optimizer=optimizer, epoch=epoch+1, 
                        hyperparams=hyperparams, checkpoint_dir=checkpoint_dir)
# endregion


# ======================= 数据加载与训练流程 =======================
# region 数据加载与训练流程
def get_vqvae_dataloader(batch_size=128):
    """
    获取用于训练VQ-VAE的MNIST数据集DataLoader
    
    参数:
        batch_size: 批量大小
    
    返回:
        (train_loader, test_loader): 训练和测试数据加载器
    """
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def get_prior_dataloader(vqvae, dataloader, device):
    """
    用训练好的VQ-VAE将原始数据转为PixelCNN Prior训练所需的indices序列
    
    参数:
        vqvae: 训练好的VQ-VAE模型
        dataloader: 原始数据加载器
        device: 计算设备
    
    返回:
        prior_loader: 一个新的DataLoader，数据为indices张量
    """
    vqvae.eval()
    all_indices = []
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            # 考虑DataParallel包装情况
            if isinstance(vqvae, nn.DataParallel):
                z_e = vqvae.module.encoder(data)
                _, indices = vqvae.module.vq(z_e)
            else:
                z_e = vqvae.encoder(data)
                _, indices = vqvae.vq(z_e)
                
            h, w = z_e.shape[2], z_e.shape[3]
            indices = indices.view(data.shape[0], h, w)  # [batch, h, w]
            all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices, dim=0)  # [N, h, w]

    # 构造TensorDataset和DataLoader
    prior_dataset = torch.utils.data.TensorDataset(all_indices)
    prior_loader = torch.utils.data.DataLoader(prior_dataset, batch_size=dataloader.batch_size, shuffle=True)
    return prior_loader


def train_vqvae_and_prior(batch_size, num_epochs, img_size, img_channel, latent_dim, num_hiddens, book_size,
                          vqvae_resume=None, prior_resume=None, checkpoint_dir='./checkpoints'):
    """
    端到端训练流程：先训练VQ-VAE，再训练PixelCNN Prior，支持从检查点恢复训练
    
    参数:
        batch_size: 批量大小
        img_size: 图像尺寸
        img_channel: 图像通道数
        latent_dim: 隐空间维度
        num_hiddens: 隐藏层维度
        book_size: 码本大小
        vqvae_resume: VQ-VAE检查点路径（可选），用于恢复训练
        prior_resume: PixelCNN Prior检查点路径（可选），用于恢复训练
        checkpoint_dir: 检查点保存目录
    """
    vqvae_train_loader, vqvae_test_loader = get_vqvae_dataloader(batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyperparams = {
        'batch_size': batch_size,
        'img_size': img_size,
        'img_channel': img_channel,
        'latent_dim': latent_dim,
        'num_hiddens': num_hiddens,
        'book_size': book_size
    }
    
    # ------------------ 训练VQ-VAE ------------------
    vqvae = VQVAE(batch_size=batch_size, 
                  img_size=img_size, 
                  img_channel=img_channel, 
                  latent_dim=latent_dim, 
                  num_hiddens=num_hiddens, 
                  book_size=book_size)
    
    train_vqvae(vqvae, vqvae_train_loader, vqvae_test_loader, 
               num_epochs=num_epochs, device=device, checkpoint_dir=checkpoint_dir,
               resume_from=vqvae_resume)
    
    # 保存最终的VQVAE模型（不仅是检查点）
    final_vqvae_path = f"{checkpoint_dir}/vqvae_final.pt"
    torch.save(vqvae.state_dict(), final_vqvae_path)
    print(f"最终VQVAE模型已保存至 {final_vqvae_path}")
    
    # ------------------ 训练PixelCNN Prior ------------------
    pixelcnn = PixelCNN(book_size=book_size, latent_dim=latent_dim, n_layers=8)
    
    # 使用训练好的VQ-VAE进行特征提取
    prior_train_loader = get_prior_dataloader(vqvae, vqvae_train_loader, device)
    prior_test_loader = get_prior_dataloader(vqvae, vqvae_test_loader, device)
    
    train_vqvae_prior(pixelcnn, vqvae, prior_train_loader, 
                     num_epochs=num_epochs, device=device, checkpoint_dir=checkpoint_dir,
                     resume_from=prior_resume)
    
    # 保存最终的Prior模型（不仅是检查点）
    final_prior_path = f"{checkpoint_dir}/pixelcnn_final.pt"
    torch.save(pixelcnn.state_dict(), final_prior_path)
    print(f"最终PixelCNN模型已保存至 {final_prior_path}")
    
    return vqvae, pixelcnn
# endregion


# ======================= 图像生成函数 =======================
# region 图像生成函数
def generate_new_images(num_samples=16, vqvae_path=None, prior_path=None, device="cuda"):
    """
    使用训练好的模型生成新图像
    
    参数:
        num_samples: 要生成的样本数量
        vqvae_path: VQ-VAE模型路径（可以是检查点或纯模型参数）
        prior_path: PixelCNN Prior模型路径（可以是检查点或纯模型参数）
        device: 计算设备
    
    返回:
        生成的图像张量，形状为[num_samples, img_channel, img_size, img_size]
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    
    # 创建模型
    # 先加载检查点获取超参数
    default_hyperparams = {
        'batch_size': num_samples,
        'img_size': 28,
        'img_channel': 1,
        'latent_dim': 128,
        'num_hiddens': 128,
        'book_size': 1024
    }
    checkpoint = torch.load(vqvae_path, map_location=device)
    hyperparams = checkpoint.get('hyperparams', default_hyperparams)

    vqvae = VQVAE(batch_size=num_samples, 
                 img_size=hyperparams['img_size'], 
                 img_channel=hyperparams['img_channel'], 
                 latent_dim=hyperparams['latent_dim'], 
                 num_hiddens=hyperparams['num_hiddens'], 
                 book_size=hyperparams['book_size'])
    prior = PixelCNN(book_size=hyperparams['book_size'], 
                    latent_dim=hyperparams['latent_dim'], 
                    n_layers=8)
    # 若有多卡，包裹DataParallel
    if torch.cuda.device_count() > 1:
        vqvae = nn.DataParallel(vqvae)
        prior = nn.DataParallel(prior)
    vqvae = vqvae.to(device)
    prior = prior.to(device)
    # 加载模型参数
    try:
        if vqvae_path.endswith('.pt'):
            try:
                load_checkpoint(vqvae_path, vqvae=vqvae, device=device)
            except:
                if isinstance(vqvae, nn.DataParallel):
                    vqvae.module.load_state_dict(torch.load(vqvae_path, map_location=device))
                else:
                    vqvae.load_state_dict(torch.load(vqvae_path, map_location=device))
                print(f"已加载VQ-VAE模型参数 {vqvae_path}")
    except Exception as e:
        print(f"加载VQ-VAE失败: {e}")
        return None
    try:
        if prior_path.endswith('.pt'):
            try:
                load_checkpoint(prior_path, pixelcnn=prior, device=device)
            except:
                if isinstance(prior, nn.DataParallel):
                    prior.module.load_state_dict(torch.load(prior_path, map_location=device))
                else:
                    prior.load_state_dict(torch.load(prior_path, map_location=device))
                print(f"已加载Prior模型参数 {prior_path}")
    except Exception as e:
        print(f"加载Prior失败: {e}")
        return None
    vqvae.eval()
    prior.eval()
    
    # 获取潜在空间尺寸
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 28, 28).to(device)
        # 考虑DataParallel包装
        if isinstance(vqvae, nn.DataParallel):
            z_e = vqvae.module.encoder(dummy)
        else:
            z_e = vqvae.encoder(dummy)
        h, w = z_e.shape[2], z_e.shape[3]
    
    # 自回归采样
    with torch.no_grad():
        # 初始化为全零
        indices = torch.zeros(num_samples, h, w).long().to(device)
        
        # 逐像素生成
        for i in range(h):
            for j in range(w):
                # 获取当前indices的预测概率
                logits = prior(indices)  # [num_samples, book_size, h, w]
                
                # 取出当前位置的logits
                logits_ij = logits[:, :, i, j]  # [num_samples, book_size]
                
                # 计算类别概率分布
                probs = F.softmax(logits_ij, dim=1)
                
                # 从分布中采样
                sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
                
                # 更新indices
                indices[:, i, j] = sampled_indices
        
        # 将indices转换为z_q，然后解码
        flat_indices = indices.reshape(-1)
        
        # 考虑DataParallel包装
        if isinstance(vqvae, nn.DataParallel):
            z_q = vqvae.module.vq.codebook(flat_indices)
            batch_size = indices.shape[0]
            feature_size = h  # h=w
            z_q = z_q.view(batch_size, feature_size, feature_size, vqvae.module.vq.latent_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            generated = vqvae.module.decoder(z_q)
        else:
            z_q = vqvae.vq.codebook(flat_indices)
            batch_size = indices.shape[0]
            feature_size = h  # h=w
            z_q = z_q.view(batch_size, feature_size, feature_size, vqvae.vq.latent_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            generated = vqvae.decoder(z_q)
    
    # 可视化生成的图像
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('generated_images.png')
    plt.close()
    
    print(f"生成的图像已保存至 'generated_images.png'")
    return generated
# endregion


# ======================= 主函数入口 =======================
if __name__ == '__main__':
    # 超参数设置
    num_epochs = 100
    batch_size = 128
    img_size = 28
    img_channel = 1
    num_hiddens = 512
    latent_dim = 512
    book_size = 64
    checkpoint_dir = './checkpoints'
    
    # 训练VQ-VAE和PixelCNN Prior，可选从检查点恢复
    vqvae_resume = './checkpoints/checkpoint_epoch100.pt' 
    prior_resume = None  # './checkpoints/checkpoint_epoch15.pt' 如果要恢复训练
    
    vqvae, pixelcnn = train_vqvae_and_prior(
        batch_size=batch_size,
        num_epochs=num_epochs,
        img_size=img_size,
        img_channel=img_channel,
        latent_dim=latent_dim,
        num_hiddens=num_hiddens,
        book_size=book_size,
        vqvae_resume=vqvae_resume,
        prior_resume=prior_resume,
        checkpoint_dir=checkpoint_dir
    )
    
    # 生成新图像
    generate_new_images(
        num_samples=16, 
        vqvae_path=f'{checkpoint_dir}/vqvae_final.pt',
        prior_path=f'{checkpoint_dir}/pixelcnn_final.pt',
        device="cuda" if torch.cuda.is_available() else "cpu"
    )