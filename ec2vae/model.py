import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import pretty_midi as pm
import numpy as np


class EC2VAE(nn.Module):

    def __init__(self,
                 roll_dims,
                 hidden_dims,
                 rhythm_dims,
                 condition_dims,
                 z1_dims,
                 z2_dims,
                 n_step,
                 k=1000,
                 device=None):
        super(EC2VAE, self).__init__()
        self.gru_0 = nn.GRU(
            roll_dims + condition_dims,
            hidden_dims,
            batch_first=True,
            bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z1_dims + z2_dims)
        self.grucell_0 = nn.GRUCell(z2_dims + rhythm_dims,
                                    hidden_dims)
        self.grucell_1 = nn.GRUCell(
            z1_dims + roll_dims + rhythm_dims + condition_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_init_0 = nn.Linear(z2_dims, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, rhythm_dims)
        self.linear_init_1 = nn.Linear(z1_dims, hidden_dims)
        self.linear_out_1 = nn.Linear(hidden_dims, roll_dims)
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.rhythm_dims = rhythm_dims
        self.sample = None
        self.rhythm_sample = None
        self.iteration = 0
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.k = torch.FloatTensor([k])
        self.device = device

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encoder(self, x, condition, sample=False):
        """
        :param x: (bs, 32, 130)
        :param condition: (bs, 32, 12)
        :param sample: bool
        :return: zp: (bs, 128), zr: (bs, 128)
        """
        self.eval()
        with torch.no_grad():
            x = torch.cat((x, condition), -1)
            x = self.gru_0(x)[-1]
            x = x.transpose_(0, 1).contiguous()
            x = x.view(x.size(0), -1)
            mu = self.linear_mu(x)
            var = self.linear_var(x).exp_()
            distribution_1 = Normal(mu[:, :self.z1_dims], var[:, :self.z1_dims])
            distribution_2 = Normal(mu[:, self.z1_dims:], var[:, self.z1_dims:])
            zp = distribution_1.sample() if sample else distribution_1.mean
            zr = distribution_2.sample() if sample else distribution_2.mean

        return zp, zr

    def rhythm_decoder(self, zr):
        self.eval()
        with torch.no_grad():
            out = torch.zeros((zr.size(0), self.rhythm_dims))
            out[:, -1] = 1.
            x = []
            t = torch.tanh(self.linear_init_0(zr))
            hx = t
            if torch.cuda.is_available():
                out = out.cuda()
            for i in range(self.n_step):
                out = torch.cat([out, zr], 1)
                hx = self.grucell_0(out, hx)
                out = F.log_softmax(self.linear_out_0(hx), 1)
                x.append(out)
                if self.training:
                    p = torch.rand(1).item()
                    if p < self.eps:
                        out = self.rhythm_sample[:, i, :]
                    else:
                        out = self._sampling(out)
                else:
                    out = self._sampling(out)
        return torch.stack(x, 1)

    def final_decoder(self, zp, rhythm, condition):
        self.eval()
        with torch.no_grad():
            out = torch.zeros((zp.size(0), self.roll_dims))
            out[:, -1] = 1.
            x, hx = [], [None, None]
            t = torch.tanh(self.linear_init_1(zp))
            hx[0] = t
            if torch.cuda.is_available():
                out = out.cuda()
            for i in range(self.n_step):
                out = torch.cat([out, rhythm[:, i, :], zp, condition[:, i, :]], 1)
                hx[0] = self.grucell_1(out, hx[0])
                if i == 0:
                    hx[1] = hx[0]
                hx[1] = self.grucell_2(hx[0], hx[1])
                out = F.log_softmax(self.linear_out_1(hx[1]), 1)
                x.append(out)
                if self.training:
                    p = torch.rand(1).item()
                    if p < self.eps:
                        out = self.sample[:, i, :]
                    else:
                        out = self._sampling(out)
                    self.eps = self.k / \
                               (self.k + torch.exp(self.iteration / self.k))
                else:
                    out = self._sampling(out)
        return torch.stack(x, 1)

    def decoder(self, zp, zr, condition=None):
        """
        :param zp: (bs, 128)
        :param zr: (bs, 128)
        :param condition: (bs, 32, 12)
        :return: (bs, 32)
        """
        self.eval()
        with torch.no_grad():
            rhythm = self.rhythm_decoder(zr)
            recon = self.final_decoder(zp, rhythm, condition)
        return recon.max(-1)[1]

    @classmethod
    def init_model(cls):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return cls(130, 2048, 3, 12, 128, 128, 32, device=device)

    def load_model(self, model_path):
        map_location = self.device
        dic = torch.load(model_path, map_location=map_location)
        for name in list(dic.keys()):
            dic[name.replace('module.', '')] = dic.pop(name)
        self.load_state_dict(dic)
        self.to(self.device)

    @staticmethod
    def note_array_to_notes(note_array, bpm, start=0.):
        notes = []
        alpha = 0.25 * 60 / bpm
        current_dur = 0
        for t in range(len(note_array) - 1, -1, -1):
            if note_array[t] == 129:
                current_dur = 0
            elif note_array[t] == 128:
                current_dur += 1
            else:
                s = start + t * alpha
                e = start + (t + current_dur + 1) * alpha
                current_dur = 0
                notes.append(pm.Note(100, note_array[t], s, e))
        return notes

    @staticmethod
    def chord_to_notes(c, bpm, start=0.):
        notes = []
        alpha = 0.25 * 60 / bpm
        current_dur = 0
        for t in range(c.shape[0] - 1, -1, -1):
            if (c[t] == 0).all():
                current_dur = 0
            else:
                if t != 0 and (c[t] == c[t - 1]).all():
                    current_dur += 1
                else:
                    s = start + t * alpha
                    e = start + (t + current_dur + 1) * alpha
                    pitches = np.where(c[t])[0]
                    notes += [pm.Note(80, p + 48, s, e) for p in pitches]
                    current_dur = 0
        return notes



