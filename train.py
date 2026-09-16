"""Train the edge CNN on MNIST and dump the fp32 weights + a raw test set.

    .venv/bin/python train.py            # ~30s on Apple Silicon / CPU
Outputs: data/model_fp32.pt, data/mnist_test.bin (u32 n, then n x [u8 label, 784 u8 pixels])
"""
import os, struct, time
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MEAN, STD = 0.1307, 0.3081


class EdgeNet(nn.Module):
    """conv-relu-pool x2 -> fc-relu -> fc. Small on purpose: this is an edge model."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def loaders(batch=128):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train = datasets.MNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA, train=False, download=True, transform=tf)
    return (torch.utils.data.DataLoader(train, batch, shuffle=True),
            torch.utils.data.DataLoader(test, 1000))


def accuracy(model, loader, device):
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(loader.dataset)


def main():
    os.makedirs(DATA, exist_ok=True)
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train, test = loaders()
    model = EdgeNet().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    t0 = time.time()
    for epoch in range(2):
        model.train()
        for x, y in train:
            opt.zero_grad()
            F.cross_entropy(model(x.to(device)), y.to(device)).backward()
            opt.step()
        print(f"epoch {epoch}: test acc {accuracy(model, test, device):.4f}  ({time.time()-t0:.0f}s)")
    torch.save(model.cpu().state_dict(), os.path.join(DATA, "model_fp32.pt"))

    # raw test set for the C++ runtime (un-normalized u8; the runtime normalizes)
    raw = datasets.MNIST(DATA, train=False, download=True)
    with open(os.path.join(DATA, "mnist_test.bin"), "wb") as f:
        f.write(struct.pack("<I", len(raw)))
        for img, label in raw:
            f.write(bytes([label]) + img.tobytes())
    print("wrote data/model_fp32.pt and data/mnist_test.bin")


if __name__ == "__main__":
    main()
