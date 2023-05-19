'''IMPORTS'''
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

'''FUNCTIONS'''
def script_evaluate(model, dataloader): 
    model.eval()
    bar = tqdm(dataloader['test'])
    acc = []
    with torch.no_grad():
        for batch in bar:
            X, y = batch
            # desnormalizar
            X = (X*0.3081 + 0.1307)*255
            # quitar dimensión canales
            X = X.squeeze(1)
            # el modelo pre-procesa
            y_hat, label = model(X)
            acc.append((y == label).sum().item() / len(y))
            bar.set_description(f"acc {np.mean(acc):.5f}")

# preparamos los datos

dataloader = {
    'train': torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ])
                      ), batch_size=2048, shuffle=True, pin_memory=True),
    'test': torch.utils.data.DataLoader(torchvision.datasets.MNIST('../data', train=False,
                   transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.1307,), (0.3081,))
                        ])
                     ), batch_size=2048, shuffle=False, pin_memory=True)
}

loaded_model = torch.jit.load('docs/model.zip')
script_evaluate(loaded_model, dataloader)