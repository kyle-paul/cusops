from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def getData():
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('/workspace/dataset', 
                            train=True, download=True, 
                            transform=transform)
    train_loader = DataLoader(dataset, batch_size=60, shuffle=True, drop_last=True)
    return train_loader