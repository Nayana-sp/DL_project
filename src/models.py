"""Model definitions. Image CNN now; audio CNN and fusion models are added in later phases."""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class ImageCNN(nn.Module):
    """Small CNN for face crops. (B,3,H,W) -> embedding (B,emb_dim) -> logits (B,n_classes)."""

    def __init__(self, n_classes=8, emb_dim=128, dropout=0.3, widths=(32, 64, 128, 256)):
        super().__init__()
        chans = (3,) + tuple(widths)
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                   nn.Linear(widths[-1], emb_dim), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, n_classes))

    def forward_embedding(self, x):
        return self.embed(self.pool(self.features(x)))

    def forward(self, x):
        return self.classifier(self.forward_embedding(x))


class ClipImageModel(nn.Module):
    """Clip-level wrapper. (B,n,3,H,W) -> clip logits (B,n_classes), the mean of per-frame logits."""

    def __init__(self, n_classes=8, emb_dim=128, dropout=0.3):
        super().__init__()
        self.cnn = ImageCNN(n_classes, emb_dim, dropout)

    def forward_embedding(self, x):
        b, n = x.shape[:2]
        return self.cnn.forward_embedding(x.flatten(0, 1)).view(b, n, -1).mean(1)

    def forward(self, x):
        b, n = x.shape[:2]
        return self.cnn(x.flatten(0, 1)).view(b, n, -1).mean(1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ClipImageModel()
    x = torch.randn(4, 5, 3, 112, 112)
    print("clip logits", tuple(m(x).shape), "| clip embedding", tuple(m.forward_embedding(x).shape),
          "| trainable params", count_params(m))