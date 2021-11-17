import torch
from torch import nn
from .dl_modules import ChordEncoder, ChordDecoder, \
    TextureEncoder, PianoTreeDecoder


class PolyDisVAE(nn.Module):

    def __init__(self, device, chd_encoder, txt_encoder, pnotree_decoder,
                 chd_decoder):
        super(PolyDisVAE, self).__init__()
        self.device = device
        self.chd_encoder = chd_encoder
        self.txt_encoder = txt_encoder
        self.pnotree_decoder = pnotree_decoder
        self.chd_decoder = chd_decoder

    def chd_encode(self, c, sample=False):
        """
        :param c: (bs, 8, 36)
        :param sample: bool
        :return: z_chd: (bs, 256)
        """
        self.eval()
        with torch.no_grad():
            dist_chd = self.chd_encoder(c)
            z_chd = dist_chd.sample() if sample else dist_chd.mean
        return z_chd

    def txt_encode(self, pr_mat, sample=False):
        """
        :param pr_mat: (bs, 32, 128)
        :param sample: bool
        :return: z_txt: (bs, 256))
        """
        self.eval()
        with torch.no_grad():
            dist_txt = self.txt_encoder(pr_mat)
            z_txt = dist_txt.sample() if sample else dist_txt.mean
        return z_txt

    def chd_decode(self, z_chd):
        """
        :param z_chd: (bs, 256)
        :return: (bs, 8, 12), (bs, 8, 12, 2), (bs, 8, 12)
        """
        self.eval()
        with torch.no_grad():
            root_pred, chroma_pred, bass_pred = \
                self.chd_decoder(z_chd, True, 0., None)
        return root_pred, chroma_pred, bass_pred

    def pnotree_decode(self, z_chd, z_txt):
        """
        :param z_chd: (bs, 256)
        :param z_txt: (bs, 256)
        :return: (bs, 32, 15, 6)
        """
        self.eval()
        with torch.no_grad():
            z = torch.cat([z_chd, z_txt], dim=-1)
            pitch_outs, dur_outs = \
                self.pnotree_decoder(z, True, None, None, 0., 0.)
            pnotree_pred, _, _ = \
                self.pnotree_decoder.output_to_numpy(pitch_outs, dur_outs)
            return pnotree_pred

    def pnotree_to_notes(self, pnotree, bpm, start=0.):
        return \
            self.pnotree_decoder.grid_to_pr_and_notes(pnotree, bpm, start)[1]

    @classmethod
    def init_model(cls, device=None, chd_size=256, txt_size=256,
                   num_channel=10):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu')
        chd_encoder = ChordEncoder(36, 1024, chd_size)

        txt_encoder = TextureEncoder(256, 1024, txt_size, num_channel)

        chd_decoder = ChordDecoder(z_dim=chd_size)

        pnotree_decoder = PianoTreeDecoder(note_embedding=None,
                                           dec_dur_hid_size=64,
                                           z_size=chd_size + txt_size)

        model = cls(device, chd_encoder, txt_encoder,
                    pnotree_decoder, chd_decoder)

        return model.to(device)

    def load_model(self, model_path):
        dic = torch.load(model_path, map_location=self.device)
        for name in list(dic.keys()):
            dic[name.replace('rhy_encoder', 'txt_encoder')] = dic.pop(name)
        for name in list(dic.keys()):
            dic[name.replace('chd_decoder', 'xxyy')] = dic.pop(name)
        for name in list(dic.keys()):
            dic[name.replace('decoder', 'pnotree_decoder')] = dic.pop(name)
        for name in list(dic.keys()):
            dic[name.replace('xxyy', 'chd_decoder')] = dic.pop(name)
        self.load_state_dict(dic)
        self.to(self.device)
