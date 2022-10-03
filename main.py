import os
import torch
import random
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F

from inference import inference
from dataset.thumos_feature import ThumosFeature
from model.model import WSTAL
from utils.loss import AttLoss, NormalizedCrossEntropy
from config.config import Config, parse_args

np.set_printoptions(formatter={'float_kind': "{:.2f}".format})


def load_weight(net, config):
    if config.load_weight:
        model_file = os.path.join(config.model_path, "model.pkl")
        print("loading from file for training: ", model_file)
        net.load_state_dict(torch.load(model_file), strict=False)


def get_dataloaders(config):
    train_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='train',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='random', supervision='strong'),
        batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers)

    test_loader = data.DataLoader(
        ThumosFeature(data_path=config.data_path, mode='test',
                      modal=config.modal, feature_fps=config.feature_fps,
                      num_segments=config.num_segments, len_feature=config.len_feature,
                      seed=config.seed, sampling='uniform', supervision='strong'),
        batch_size=1,
        shuffle=False, num_workers=config.num_workers)

    return train_loader, test_loader


def set_seed(config):
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False


class ThumosTrainer():
    def __init__(self, config):
        # config
        self.config = config

        # network
        self.net = WSTAL(config)
        self.net = self.net.cuda()

        # data
        self.train_loader, self.test_loader = get_dataloaders(self.config)

        # loss, optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.lr, betas=(0.9, 0.999),
                                          weight_decay=0.0005)

        self.loss_att = AttLoss(8)
        self.loss_nce = NormalizedCrossEntropy()

        # parameters
        self.best_mAP = -1  # init
        self.step = 0
        self.total_loss_per_epoch = 0

    def test(self):
        self.net.eval()

        with torch.no_grad():
            model_filename = "model.pkl"
            self.config.model_file = os.path.join(self.config.model_path, model_filename)
            _mean_ap, test_acc, ap = inference(self.net, self.config, self.test_loader,
                                               model_file=self.config.model_file)
            print("cls_acc={:.5f} map={:.5f}".format(test_acc * 100, _mean_ap * 100))

            print('map[0.1:0.5] = %0.2f' % (np.sum(ap[:5]) * 100 / 5))
            print('map[0.3:0.7] = %0.2f' % (np.sum(ap[2:7]) * 100 / 5))

    def evaluate(self, epoch=0):
        if self.step % self.config.detection_inf_step == 0:
            self.total_loss_per_epoch /= self.config.detection_inf_step

            with torch.no_grad():
                self.net = self.net.eval()
                self.net = self.net.eval()
                mean_ap, test_acc, ap = inference(self.net, self.config, self.test_loader, model_file=None)
                self.net = self.net.train()

                f_path = os.path.join(self.config.model_path, 'step: {}.txt'.format(self.step))
                with open(f_path, 'w') as f:
                    iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    string_to_write = "step: {}  mAP: {:.2f}".format(self.step, mean_ap * 100)
                    f.write(string_to_write + '\n')
                    f.flush()

                    sum = 0
                    count = 0
                    for item in list(zip(iou, ap)):
                        sum = sum + item[1]
                        count += 1
                        string_to_write = 'map @ %0.1f = %0.2f' % (item[0], item[1] * 100)
                        f.write(string_to_write + '\n')
                        f.flush()
                    string_to_write = 'map[0.1:0.5] = %0.2f' % (np.sum(ap[:5]) * 100 / 5)
                    f.write(string_to_write + '\n')
                    f.flush()
                    string_to_write = 'map[0.3:0.7] = %0.2f' % (np.sum(ap[2:7]) * 100 / 5)
                    f.write(string_to_write + '\n')
                    f.flush()
                    string_to_write = 'map[0.1:0.7] = %0.2f' % (np.sum(ap[:7]) * 100 / 7)
                    f.write(string_to_write + '\n')
                    f.flush()

            if mean_ap > self.best_mAP:
                self.best_mAP = mean_ap
                torch.save(self.net.state_dict(), os.path.join(self.config.model_path, "CAS_Only.pkl"))

                f_path = os.path.join(self.config.model_path, 'best.txt')
                with open(f_path, 'w') as f:
                    iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    string_to_write = "step: {}  mAP: {:.2f}".format(self.step, mean_ap * 100)
                    f.write(string_to_write + '\n')
                    f.flush()

                    sum = 0
                    count = 0
                    for item in list(zip(iou, ap)):
                        sum = sum + item[1]
                        count += 1
                        string_to_write = 'map @ %0.1f = %0.2f' % (item[0], item[1] * 100)
                        f.write(string_to_write + '\n')
                        f.flush()
                    string_to_write = 'map[0.1:0.5] = %0.2f' % (np.sum(ap[:5]) * 100 / 5)
                    f.write(string_to_write + '\n')
                    f.flush()
                    string_to_write = 'map[0.3:0.7] = %0.2f' % (np.sum(ap[2:7]) * 100 / 5)
                    f.write(string_to_write + '\n')
                    f.flush()
                    string_to_write = 'map[0.1:0.7] = %0.2f' % (np.sum(ap[:7]) * 100 / 7)
                    f.write(string_to_write + '\n')
                    f.flush()

            print("epoch={:5d}  step={:5d}  Loss={:.4f}  cls_acc={:5.2f}  best_map={:5.2f}".format(
                epoch, self.step, self.total_loss_per_epoch, test_acc * 100, self.best_mAP * 100))

            self.total_loss_per_epoch = 0

    def train(self):
        # resume training
        load_weight(self.net, self.config)

        # training
        for epoch in range(self.config.num_epochs):

            for _data, _label, _, _, _, sample_idxs in self.train_loader:
                _data, _label = _data.cuda(), _label.cuda()
                self.optimizer.zero_grad()

                f_labels = torch.cat((_label, torch.zeros_like(_label[..., :1])), -1)
                b_labels = torch.cat((_label, torch.ones_like(_label[..., :1])), -1)

                ############################## Baseline ##########################
                ca_vid_pred, cw_vid_pred, frm_fg_scrs, frm_scrs, frm_scrs_supp, feat_emb = self.net(_data)

                vid_fore_loss = self.loss_nce(ca_vid_pred, f_labels)
                vid_back_loss = self.loss_nce(cw_vid_pred, b_labels)
                vid_att_loss = self.loss_att(frm_fg_scrs)

                vid_loss = vid_fore_loss + vid_back_loss + vid_att_loss * self.config.lambda_a

                total_loss = vid_loss

                ##############################  C3BN  ##########################

                B = _data.size()[0]
                T = _data.size()[1]

                #### convex combination of input features
                lmda = torch.distributions.beta.Beta(2, 2).sample((B, T, 1)).cuda()
                lmda = lmda[:, :-1]
                _data1 = lmda * _data[:, 1:] + (1 - lmda) * _data[:, :-1]

                #### compute video loss for child snippets
                ca_vid_pred1, cw_vid_pred1, frm_fg_scrs1, frm_scrs1, frm_scrs_supp1, feat_emb1 = self.net(_data1)
                vid_fore_loss1 = self.loss_nce(ca_vid_pred1, f_labels)
                vid_back_loss1 = self.loss_nce(cw_vid_pred1, b_labels)
                vid_att_loss1 = self.loss_att(frm_fg_scrs1)
                vid_loss1 = vid_fore_loss1 + vid_back_loss1 + vid_att_loss1 * self.config.lambda_a
                total_loss += vid_loss1 * self.config.lambda_1

                #### compute consistency loss
                frm_scrs1 = frm_scrs1.softmax(-1)
                frm_scrs = frm_scrs.softmax(-1)
                frm_scrs_s1 = lmda * frm_scrs[:, 1:] + (1 - lmda) * frm_scrs[:, :-1]
                consistency_loss = (((frm_scrs1 - frm_scrs_s1) ** 2).sum(1)[b_labels > 0]).mean()
                total_loss += consistency_loss * self.config.lambda_2

                #### compute contrastive loss
                sample_idxs = sample_idxs.float().cuda()
                pos_emb = torch.eye(T).float().cuda()
                pos_emb = pos_emb[None].repeat(B, 1, 1)
                pos_emb1 = lmda * pos_emb[:, 1:] + (1 - lmda) * pos_emb[:, :-1]
                pos_sim = torch.bmm(pos_emb, pos_emb1.transpose(2, 1))
                sample_idxs1 = lmda[..., 0] * sample_idxs[:, 1:] + (1 - lmda[..., 0]) * sample_idxs[:, :-1]
                idx_sim = (sample_idxs[:, :, None] != sample_idxs1[:, None, :]).float()
                feat_sim = torch.bmm(feat_emb, feat_emb1.transpose(2, 1)).div(0.1)
                feat_sim = feat_sim * idx_sim
                pos_sim = pos_sim * idx_sim
                feat_sim0 = feat_sim.log_softmax(dim=-1)
                pos_sim0 = F.normalize(pos_sim, dim=-1, p=1)
                contrastive_loss = -((feat_sim0 * pos_sim0).sum(-1)).mean()
                feat_sim1 = feat_sim.transpose(2, 1).log_softmax(dim=-1)
                pos_sim1 = F.normalize(pos_sim.transpose(2, 1), dim=-1, p=1)
                contrastive_loss += -((feat_sim1 * pos_sim1).sum(-1)).mean()
                total_loss += contrastive_loss * self.config.lambda_3

                total_loss.backward()
                self.optimizer.step()

                self.total_loss_per_epoch += total_loss.cpu().item()
                self.step += 1

                # evaluation
                self.evaluate(epoch=epoch)


def main():
    args = parse_args()
    config = Config(args)
    set_seed(config)

    trainer = ThumosTrainer(config)

    if args.inference_only:
        trainer.test()
    else:
        trainer.train()


if __name__ == '__main__':
    main()
