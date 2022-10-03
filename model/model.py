import torch
import torch.nn as nn
import torch.nn.functional as F


class WSTAL(nn.Module):
    def __init__(self, args):
        super().__init__()
        # feature embedding
        self.n_in = args.len_feature
        self.n_out = 1024
        self.n_class = args.num_classes
        self.arg = args

        self.base_module = nn.Sequential(
            nn.Conv1d(in_channels=self.n_in, out_channels=self.n_out, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.cas_module = nn.Conv1d(in_channels=self.n_out, out_channels=self.n_class + 1, kernel_size=1, padding=0)
        self.att_module = nn.Conv1d(in_channels=self.n_out, out_channels=1, kernel_size=1, padding=0)

        self.proj = nn.Conv1d(in_channels=self.n_out, out_channels=128, kernel_size=1, padding=0)

    def forward(self, x):
        input = x.permute(0, 2, 1)
        base = self.base_module(input)
        frm_logits = self.cas_module(base).permute(0, 2, 1)
        att = self.att_module(base).sigmoid().squeeze(1)
        base_supp = base * att.unsqueeze(1)
        frm_logits_supp = self.cas_module(base_supp).permute(0, 2, 1)

        ca_vid_pred = torch.topk(frm_logits, self.arg.num_segments // 8, dim=1)[0].mean(1)
        ca_vid_pred = F.softmax(ca_vid_pred, dim=-1)

        cw_vid_pred = torch.topk(frm_logits_supp, self.arg.num_segments // 8, dim=1)[0].mean(1)
        cw_vid_pred = F.softmax(cw_vid_pred, dim=-1)

        if self.training:
            emb = self.proj(base).permute(0, 2, 1)
            emb = F.normalize(emb, dim=-1, p=2)
            return ca_vid_pred, cw_vid_pred, att, frm_logits, frm_logits_supp, emb
        else:
            return ca_vid_pred, cw_vid_pred, att, frm_logits
