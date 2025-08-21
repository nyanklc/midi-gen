import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

from data import get_processed_dataset, split_dataset, visualize_mat

TIME_LEN = 5000
BATCH_SIZE = 16
NR_EPOCHS = 1
TRAIN_RATIO = 0.1
OUT_DIR = "./out/"


def roll_to_midi(roll, fs=16):
    from symusic import Score, Part, Note
    """
    roll: np.array shape (88, T), values 0/1
    fs: frames per second (time resolution)
    """
    score = Score()
    part = Part("Piano")

    for pitch in range(88):
        start = None
        for t in range(roll.shape[1]):
            if roll[pitch, t] == 1:
                if start is None:
                    start = t
            else:
                if start is not None:
                    # convert to ticks
                    note = Note(
                        start=int(start * (score.ticks_per_quarter / fs)),
                        end=int(t * (score.ticks_per_quarter / fs)),
                        pitch=pitch + 21,   # MIDI pitch (A0=21)
                        velocity=64
                    )
                    part.notes.append(note)
                    start = None
        # if note still active at end
        if start is not None:
            note = Note(
                start=int(start * (score.ticks_per_quarter / fs)),
                end=int(roll.shape[1] * (score.ticks_per_quarter / fs) / fs),
                pitch=pitch + 21,
                velocity=64
            )
            part.notes.append(note)

    score.parts.append(part)
    return score

def binarize_roll(roll, threshold=0.5):
    return (roll > threshold).astype(np.uint8)  # 0/1 piano roll


class Generator(nn.Module):
    def __init__(self, z_dim=128, fixed_len=TIME_LEN):
        super().__init__()
        self.fixed_len = fixed_len

        # project and reshape
        self.fc = nn.Linear(z_dim, 256 * (88//8) * (fixed_len//8))

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # outputs in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 256, 88//8, self.fixed_len//8)
        return self.net(x)


import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class Discriminator(nn.Module):
    def __init__(self, fixed_len=TIME_LEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            ResBlock(64, 128, stride=2),
            ResBlock(128, 256, stride=2),
            ResBlock(256, 512, stride=2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cuda device: {torch.cuda.get_device_name(torch.cuda.device)}")
    print("------------------------------------------------------------")

    dataset = get_processed_dataset(TIME_LEN)
    print(f"dataset data shape and type: {dataset.__getitem__(0).shape}, {dataset.__getitem__(0).dtype}")

    train_set, test_set = None, None
    if TRAIN_RATIO == 1:
        train_set = dataset
    else:
        train_set, test_set = split_dataset(dataset, (TRAIN_RATIO, 1 - TRAIN_RATIO))
    print(f"training set length: {len(train_set) if train_set is not None else "-"}")
    print(f"testing set length: {len(test_set) if test_set is not None else "-"}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


    ### TEST ###

    z_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(z_dim=z_dim, fixed_len=TIME_LEN).to(device)
    D = Discriminator(fixed_len=TIME_LEN).to(device)

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    epochs = NR_EPOCHS
    for epoch in range(epochs):
        iter = 0
        for real in train_loader:
            print(f"iteration {iter} of {len(train_loader)}")
            real = real.to(device)            # (B,1,88,fixed_len)
            real = real.unsqueeze(1)          # (B, 1, 88, T)
            B = real.size(0)

            # Labels
            real_labels = torch.ones(B, 1, device=device)
            fake_labels = torch.zeros(B, 1, device=device)

            ## ---- Train Discriminator ----
            print("-- Train D")
            z = torch.randn(B, z_dim, device=device)
            fake = G(z).detach()

            out_real = D(real)
            out_fake = D(fake)

            loss_D = criterion(out_real, real_labels) + criterion(out_fake, fake_labels)
            print(f"-- Calculated D loss: {loss_D.item()}")

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            print("-- Optimized D loss")

            ## ---- Train Generator ----
            print("-- Train G")
            z = torch.randn(B, z_dim, device=device)
            fake = G(z)

            out_fake = D(fake)
            loss_G = criterion(out_fake, real_labels)  # generator wants D(fake)=1
            print(f"-- Calculated G loss: {loss_G.item()}")

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            print("-- Optimized G loss")

            iter = iter + 1

        print(f"Epoch {epoch+1}, Loss_D: {loss_D.item():.4f}, Loss_G: {loss_G.item():.4f}")

    print("TRAINING COMPLETE")

    # Save
    torch.save(G.state_dict(), OUT_DIR + "generator.pth")
    torch.save(D.state_dict(), OUT_DIR + "discriminator.pth")

    print("SAVED MODELS")

    # Generate
    print("GENERATING")
    G.eval()  # set to eval mode (important for inference)
    z = torch.randn(5, z_dim, device=device)  # 5 samples
    rolls = None
    with torch.no_grad():
        rolls = G(z).cpu().numpy()  # shape (B, 1, 88, T)
        rolls = rolls[:, 0]         # shape (B, 88, T)

    for roll in rolls:
        visualize_mat(roll)


    # for i in range(rolls.shape[0]):
    #     roll = binarize_roll(rolls[i,0])
    #     midi = roll_to_midi(roll, fs=16)
    #     midi.dump_midi(OUT_DIR + f"generated_{i}.midi")


if __name__ == "__main__":
    main()
