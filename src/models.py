
import torch
from torch import nn
from torchvision import models


class AudioCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=8):
        super().__init__()

        blocks = []
        in_ch = 1

        for out_ch in (32, 64, 128):
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch

        self.features = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.embedding = nn.Sequential(
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x):
        return self.embedding(self.features(x))

    def forward(self, x):
        return self.classifier(self.encode(x))


class VisualCNN(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_classes=8,
        pretrained=True,
    ):
        super().__init__()

        try:
            weights = (
                models.ResNet18_Weights.DEFAULT
                if pretrained
                else None
            )
            backbone = models.resnet18(weights=weights)

        except Exception as exc:
            print(
                f"WARNING: pretrained ResNet weights unavailable "
                f"({exc}); using random initialization."
            )
            backbone = models.resnet18(weights=None)

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes,
        )

    def encode_frames(self, x):
        # Input shape: B, T, C, H, W
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        with torch.no_grad():
            f = self.backbone(x).flatten(1)

        f = f.reshape(b, t, -1).mean(dim=1)

        return self.projection(f)

    def forward(self, x):
        return self.classifier(self.encode_frames(x))


class MultimodalModel(nn.Module):
    def __init__(
        self,
        mode="early",
        embedding_dim=128,
        num_classes=8,
        pretrained=True,
        modality_dropout=False,
    ):
        super().__init__()

        self.mode = mode
        self.modality_dropout = modality_dropout

        self.audio = AudioCNN(embedding_dim, num_classes)
        self.visual = VisualCNN(
            embedding_dim,
            num_classes,
            pretrained,
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.visual_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        # Early fusion concatenates two embeddings:
        # embedding_dim + embedding_dim = 2 * embedding_dim.
        # Both early and gated variants need this input size.
        fusion_input_dim = (
            embedding_dim * 2
            if mode in ("early", "early_dropout")
            else embedding_dim
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )

    def forward(self, audio, video):
        if self.mode == "audio":
            return self.audio(audio)

        if self.mode == "visual":
            return self.visual(video)

        if self.mode == "late":
            return (
                self.audio(audio) + self.visual(video)
            ) / 2

        a = self.audio_proj(self.audio.encode(audio))
        v = self.visual_proj(
            self.visual.encode_frames(video)
        )

        if self.training and self.modality_dropout:
            # Randomly drop audio and/or visual embeddings
            # during training to improve robustness.
            r = torch.rand(a.shape[0], device=a.device)

            a = a * (r >= 0.25).float().unsqueeze(1)
            v = v * (r < 0.75).float().unsqueeze(1)

        if self.mode in ("early", "early_dropout"):
            return self.fusion_head(
                torch.cat([a, v], dim=1)
            )

        if self.mode in ("gated", "gated_dropout"):
            g = self.gate(torch.cat([a, v], dim=1))
            self.last_gate = g.detach()

            return self.fusion_head(
                g * a + (1 - g) * v
            )

        raise ValueError(f"Unknown model mode: {self.mode}")
