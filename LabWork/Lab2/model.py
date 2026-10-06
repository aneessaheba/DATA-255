import torch.nn as nn

class RB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1))

    def forward(self, x):
        return x + self.b(x)

class TriScaleSR(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.head = nn.Conv2d(3, c, 3, 1, 1)
        self.e1 = nn.Sequential(RB(c), RB(c))
        self.d1 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, 2, 1), nn.ReLU(True))
        self.e2 = nn.Sequential(RB(2 * c), RB(2 * c))
        self.d2 = nn.Sequential(nn.Conv2d(2 * c, 4 * c, 3, 2, 1), nn.ReLU(True))
        self.mid = nn.Sequential(*[RB(4 * c) for _ in range(6)])
        self.u2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, 2)
        self.f2 = nn.Sequential(RB(2 * c), RB(2 * c))
        self.u1 = nn.ConvTranspose2d(2 * c, c, 2, 2)
        self.f1 = nn.Sequential(RB(c), RB(c))
        self.tail = nn.Conv2d(c, 3, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        f = self.head(x)
        a = self.e1(f)
        b = self.e2(self.d1(a))
        y = self.f2(b + self.u2(self.mid(self.d2(b))))
        y = self.f1(a + self.u1(y))
        return x + self.tail(y + f)
