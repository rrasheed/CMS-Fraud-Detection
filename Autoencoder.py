import torch.nn as nn

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                                      SHALLOW AE
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class ShallowAE(nn.Module):  # AE_2HL(input_size, hl1)
    def __init__(self, input_size, hl1):
        super(ShallowAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hl1),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hl1, input_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                              DEEP AE - 2 HIDDEN LAYERS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class AE_2HL(nn.Module):  # AE_2HL(input_size, hl1, hl2)
    def __init__(self, input_size, hl1, hl2):
        super(AE_2HL, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hl1),
            nn.Tanh(),
            nn.Linear(hl1, hl2),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hl2, hl1),
            nn.Tanh(),
            nn.Linear(hl1, input_size),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                              DEEP AE - 3 HIDDEN LAYERS
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


class AE_3HL(nn.Module):  # AE_2HL(input_size, hl1, hl2, hl3)
    def __init__(self, input_size, hl1, hl2, hl3):
        super(AE_3HL, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hl1),
            nn.Tanh(),
            nn.Linear(hl1, hl2),
            nn.Tanh(),
            nn.Linear(hl2, hl3),
            nn.LeakyReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hl3, hl2),
            nn.Tanh(),
            nn.Linear(hl2, hl1),
            nn.Tanh(),
            nn.Linear(hl1, input_size),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x