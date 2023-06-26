# MNIST image generation using GAN
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from PIL import Image
# Parameters
image_size = 100
G_input_dim = 100
G_output_dim = image_size*image_size
D_input_dim = image_size*image_size
D_output_dim = 1
hidden_dims = [256,256,256, 512,512,512, 1024,1024,1024]

learning_rate = 0.0002
batch_size =512
num_epochs =50
data_dir = 'dogs'
save_dir = 'MNIST_GAN_results/'

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.imgs = self._load_images()

    def _load_images(self):
        images = []
        root = self.root
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            if self.is_valid_file(path):
                item = (path, 0)  # 0 represents the class label for plane images
                images.append(item)
        return images

    def is_valid_file(self, x):
        return x.endswith('.jpg') or x.endswith('.jpeg') or x.endswith('.png')

    def __getitem__(self, index):
        path, target = self.imgs[index]
        image = Image.open(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.imgs)
# MNIST dataset
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# plane_data = dsets.ImageFolder(root=data_dir, transform=transform)
plane_data = CustomImageFolder(root=data_dir, transform=transform)

# Data yükleyici (Data Loader) oluştur
data_loader = torch.utils.data.DataLoader(dataset=plane_data,
                                          batch_size=batch_size,
                                          shuffle=True)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Generator model
class Generator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Generator, self).__init__()

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(hidden_dims)):
            # Fully-connected layer
            fc_name = 'fc' + str(i+1)
            if i == 0:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(input_dim, hidden_dims[i], bias=True))
            else:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], output_dim, bias=True),
            torch.nn.Tanh()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Discriminator model
class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Discriminator, self).__init__()

        # Hidden layer
        self.hidden_layer = torch.nn.Sequential()
        for i in range(len(hidden_dims)):
            # Fully-connected layer
            fc_name = 'fc' + str(i + 1)
            if i == 0:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(input_dim, hidden_dims[i], bias=True))
            else:
                self.hidden_layer.add_module(fc_name, torch.nn.Linear(hidden_dims[i-1], hidden_dims[i], bias=True))
            # Activation
            act_name = 'act' + str(i + 1)
            self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))
            # Dropout
            drop_name = 'drop' + str(i + 1)
            self.hidden_layer.add_module(drop_name, torch.nn.Dropout(0.3))

        # Output layer
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[i], output_dim, bias=True),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)
        return out


# Plot losses
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='MNIST_GAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_GAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_result(generator, noise, num_epoch, save=False, save_dir='MNIST_GAN_results/', show=False, fig_size=(5, 5)):
    generator.eval()

    noise = Variable(noise.cuda())
    gen_image = generator(noise)
    gen_image = denorm(gen_image)

    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.set_adjustable('box')
        ax.imshow(img.cpu().data.view(image_size, image_size).numpy(), cmap='gray', aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch+1)
    fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'MNIST_GAN_epoch_{:d}'.format(num_epoch+1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Models
G = Generator(G_input_dim, hidden_dims, G_output_dim)
D = Discriminator(D_input_dim, hidden_dims[::-1], D_output_dim)
G.cuda()
D.cuda()

# Loss function
criterion = torch.nn.BCELoss()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

# Training GAN
D_avg_losses = []
G_avg_losses = []

# Fixed noise for test
num_test_samples = 5*5
fixed_noise = torch.randn(num_test_samples, G_input_dim)

for epoch in range(num_epochs):
    D_losses = []
    G_losses = []

    # minibatch training
    for i, (images, _) in enumerate(data_loader):

        # image data
        mini_batch = images.size()[0]
        x_ = images.view(-1, D_input_dim)
        x_ = Variable(x_.cuda())

        # labels
        y_real_ = Variable(torch.ones(x_.size(0), 1).cuda())
        y_fake_ = Variable(torch.zeros(x_.size(0), 1).cuda())

        # Train discriminator with real data
        D_real_decision = D(x_)
        D_real_loss = criterion(D_real_decision, y_real_[:D_real_decision.size(0)])

        # Train discriminator with fake data
        z_ = torch.randn(mini_batch, G_input_dim)
        z_ = Variable(z_.cuda())
        gen_image = G(z_)

        D_fake_decision = D(gen_image)
        D_fake_loss = criterion(D_fake_decision, y_fake_[:D_fake_decision.size(0)])

        # Back propagation
        D_loss = D_real_loss + D_fake_loss
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        z_ = torch.randn(mini_batch, G_input_dim)
        z_ = Variable(z_.cuda())
        gen_image = G(z_)

        D_fake_decision = D(gen_image)
        D_fake_decision = D_fake_decision[:y_real_.size(0)]  # Crop D_fake_decision to match y_real_ size

        G_loss = criterion(D_fake_decision, y_fake_[:D_fake_decision.size(0)])  # Use y_fake_ instead of y_real_

        # Back propagation
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, num_epochs, i+1, len(data_loader), D_loss.item(), G_loss.item()))

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)

    plot_loss(D_avg_losses, G_avg_losses, epoch, save=True)

    # Show result for fixed noise
    plot_result(G, fixed_noise, epoch, save=True, fig_size=(5, 5))


# Make gif
loss_plots = []
gen_image_plots = []
for epoch in range(num_epochs):
    # plot for generating gif
    save_fn1 = save_dir + 'MNIST_GAN_losses_epoch_{:d}'.format(epoch + 1) + '.png'
    loss_plots.append(imageio.imread(save_fn1))

    save_fn2 = save_dir + 'MNIST_GAN_epoch_{:d}'.format(epoch + 1) + '.png'
    gen_image_plots.append(imageio.imread(save_fn2))

imageio.mimsave(save_dir + 'MNIST_GAN_losses_epochs_{:d}'.format(num_epochs) + '.gif', loss_plots, duration=200)
imageio.mimsave(save_dir + 'MNIST_GAN_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, duration=200)
