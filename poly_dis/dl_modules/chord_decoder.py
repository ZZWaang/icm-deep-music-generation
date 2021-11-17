import torch
from torch import nn
import random


class ChordDecoder(nn.Module):

    def __init__(self, input_dim=36, z_input_dim=256,
                 hidden_dim=512, z_dim=256, num_step=32):
        super(ChordDecoder, self).__init__()
        self.z2dec_hid = nn.Linear(z_dim, hidden_dim)
        self.z2dec_in = nn.Linear(z_dim, z_input_dim)
        self.gru = nn.GRU(input_dim + z_input_dim, hidden_dim,
                          batch_first=True,
                          bidirectional=False)
        self.init_input = nn.Parameter(torch.rand(36))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.root_out = nn.Linear(hidden_dim, 12)
        self.chroma_out = nn.Linear(hidden_dim, 24)
        self.bass_out = nn.Linear(hidden_dim, 12)
        self.num_step = num_step

    def forward(self, z_chd, inference, tfr, c=None):
        # z_chd: (B, z_chd_size)
        bs = z_chd.size(0)
        z_chd_hid = self.z2dec_hid(z_chd).unsqueeze(0)
        z_chd_in = self.z2dec_in(z_chd).unsqueeze(1)
        if inference:
            tfr = 0.
        token = self.init_input.repeat(bs, 1).unsqueeze(1)
        recon_root = []
        recon_chroma = []
        recon_bass = []

        for t in range(int(self.num_step / 4)):
            chd, z_chd_hid = \
                self.gru(torch.cat([token, z_chd_in], dim=-1), z_chd_hid)
            r_root = self.root_out(chd)  # (bs, 1, 12)
            r_chroma = self.chroma_out(chd).view(bs, 1, 12, 2).contiguous()
            r_bass = self.bass_out(chd)  # (bs, 1, 12)
            recon_root.append(r_root)
            recon_chroma.append(r_chroma)
            recon_bass.append(r_bass)

            t_root = torch.zeros(bs, 1, 12).to(z_chd.device).float()
            t_root[torch.arange(0, bs), 0, r_root.max(-1)[-1]] = 1.
            t_chroma = r_chroma.max(-1)[-1].float()
            t_bass = torch.zeros(bs, 1, 12).to(z_chd.device).float()
            t_bass[torch.arange(0, bs), 0, r_bass.max(-1)[-1]] = 1.
            token = torch.cat([t_root, t_chroma, t_bass], dim=-1)
            if t == self.num_step - 1:
                break
            teacher_force = random.random() < tfr
            if teacher_force and not inference:
                token = c[:, t].unsqueeze(1)
        recon_root = torch.cat(recon_root, dim=1)
        recon_chroma = torch.cat(recon_chroma, dim=1)
        recon_bass = torch.cat(recon_bass, dim=1)
        return recon_root, recon_chroma, recon_bass
