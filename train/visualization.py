import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from network import Model
from dataset import getData

# load dataset
train_loader = getData()

# load model with pretrained weights
model = Model()
model.load_state_dict(torch.load("/workspace/models/model.pt"))
model.cuda().eval()

loss_func = nn.CrossEntropyLoss()

with torch.no_grad():
    for image, label in train_loader:
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        loss = loss_func(output, label)
        print(loss.item())

        output = torch.argmax(output, dim=-1)
        image = image.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        output = output.detach().cpu().numpy()

        break



fig, axes = plt.subplots(6, 10)
for i in range(6):
    for j in range(10):
        index = i * 10 + j
        img = image[index, 0]
        gt = label[index]
        pred = output[index]
        axes[i, j].imshow(img, cmap="gray")
        axes[i, j].set_title(f'{gt} - {pred}')
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()