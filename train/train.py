import torch
import torch.nn as nn
from dataset import getData
from network import Model
from tqdm import tqdm

# Load dataset
train_loader = getData()

model = Model().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

step = len(train_loader)
min_loss = 1000

for epoch in tqdm(range(30)):
    print(f"Epoch: {epoch}")
    total_loss = 0
    for image, label in train_loader:
        image = image.cuda()
        label = label.cuda()
        output = model(image)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    cur_loss = total_loss / step
    if (cur_loss < min_loss):
        min_loss = cur_loss
        torch.save(model.state_dict(), "/workspace/models/model.pt")
        print("update weight version")

    print(f"Loss: {round(cur_loss, 5)}")