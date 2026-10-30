"""Model definitions. Image CNN now; audio CNN and fusion models are added in later phases."""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool2d(2))


class ImageCNN(nn.Module):
    """Small CNN for face crops. (B,3,H,W) -> embedding (B,emb_dim) -> logits (B,n_classes)."""

    def __init__(self, n_classes=8, emb_dim=128, dropout=0.3, widths=(32, 64, 128, 256), in_ch=3):
        super().__init__()
        chans = (in_ch,) + tuple(widths)
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

class AudioCNN(ImageCNN):
    """Same conv design as ImageCNN, for log-Mel input: (B,1,n_mels,T) -> embedding (B,emb_dim) -> logits."""

    def __init__(self, n_classes=8, emb_dim=128, dropout=0.3, widths=(32, 64, 128, 256)):
        super().__init__(n_classes, emb_dim, dropout, widths, in_ch=1)

class EarlyFusionModel(nn.Module):
    """Early (feature-level) fusion: concatenate the clip image embedding and the audio embedding.

    Input: (frames (B,n,3,H,W), log-Mel (B,1,n_mels,T)) -> logits (B,n_classes).
    """

    def __init__(self, n_classes=8, emb_dim=128, dropout=0.3):
        super().__init__()
        self.image = ClipImageModel(n_classes, emb_dim, dropout)
        self.audio = AudioCNN(n_classes, emb_dim, dropout)
        self.image.cnn.classifier = nn.Identity()  # per-branch heads are not used here
        self.audio.classifier = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * emb_dim, emb_dim), nn.ReLU(inplace=True),
                                  nn.Dropout(dropout), nn.Linear(emb_dim, n_classes))

    def forward_embedding(self, x):
        frames, spec = x
        return torch.cat([self.image.forward_embedding(frames), self.audio.forward_embedding(spec)], dim=1)

    def forward(self, x):
        return self.head(self.forward_embedding(x))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = ClipImageModel()
    x = torch.randn(4, 5, 3, 112, 112)
    print("clip logits", tuple(m(x).shape), "| clip embedding", tuple(m.forward_embedding(x).shape),
          "| trainable params", count_params(m))
    
    am = AudioCNN()
    xa = torch.randn(4, 1, 64, 501)
    print("audio logits", tuple(am(xa).shape), "| audio embedding", tuple(am.forward_embedding(xa).shape),
          "| trainable params", count_params(am))
    
    fm = EarlyFusionModel()
    xf = (torch.randn(4, 5, 3, 112, 112), torch.randn(4, 1, 64, 501))
    print("early fusion logits", tuple(fm(xf).shape), "| trainable params", count_params(fm))    