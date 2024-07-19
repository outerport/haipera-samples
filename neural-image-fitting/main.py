import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from PIL import Image

data_path = 'data/sample.jpg'

learning_rate = 0.001
num_epochs = 5
batch_size = 1024
num_layers = 2
hidden_size = 64

def coordinate_grid(h: int, w: int):
    x = torch.linspace(-1, 1, w)
    y = torch.linspace(-1, 1, h)
    grid = torch.stack(torch.meshgrid(y, x, indexing='ij'), dim=-1)
    return grid

def load_image(path):
    img = Image.open(path)
    img = np.array(img).astype(np.float32) / 255
    img = torch.tensor(img)
    return img

def save_image(img, path):
    img = img.squeeze(0).detach().cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.img = load_image(path)
        self.size = self.img.shape[:2]
        self.pixels = self.img.view(-1, 3)
        self.coords = coordinate_grid(*self.size).view(-1, 2)

    def __len__(self):
        return self.pixels.shape[0]

    def __getitem__(self, idx):
        return self.coords[idx], self.pixels[idx]
    
class ImageModel(nn.Module):
    def __init__(self, num_layers, hidden_size, input_multiplier=2.0, hidden_multiplier=3.0):
        super().__init__()
        self.output_multiplier = 2.0
        self.input_multiplier = input_multiplier
        self.fc = nn.Sequential(
            nn.Linear(2, hidden_size),
            *[nn.Sequential(
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            ) for _ in range(num_layers)],
            nn.Linear(hidden_size, 3)
        )

    def forward(self, x):
        return self.fc(x * self.input_multiplier) * self.output_multiplier
    
dataset = ImageDataset(data_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = ImageModel(num_layers, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for i in range(num_epochs):

    for coords, pixels in loader:
        optimizer.zero_grad()
        pred = model(coords)
        loss = F.mse_loss(pred, pixels)
        loss.backward()
        optimizer.step()

    print(f'Iteration {i}, Loss: {loss.item()}')
    img = model(dataset.coords)
    img = img.view(*dataset.size, 3)
    os.makedirs('output', exist_ok=True)
    save_image(img, f'output/{i}.jpg')
