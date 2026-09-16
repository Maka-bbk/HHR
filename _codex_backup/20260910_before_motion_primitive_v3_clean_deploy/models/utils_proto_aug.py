import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.trial_pooling import move_trial_batch_to_device


'''
2024-05-03
prototype augmentation utils, including:
* prototype augmentation loss functions
* save prototypes for the offline session (according to ground-truth labels)
* save prototypes for each online continual session (according to pseudo-labels)

update 1: [2024-05-04] hardness-aware prototype sampling
'''


class ProtoAugManager:
    def __init__(self, feature_dim, batch_size, hardness_temp, radius_scale, device, logger):
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.device = device
        self.prototypes = None
        self.mean_similarity = None   # NOTE!!! mean similarity of each prototype, for hardness-aware sampling
        self.hardness_temp = hardness_temp   # NOTE!!! temperature to compute mean similarity to softmax prob for hardness-aware sampling
        self.radius = 0
        self.radius_scale = radius_scale
        self.logger = logger


    def _sample_count(self, batch_size):
        sample_count = self.batch_size if batch_size is None else int(batch_size)
        if sample_count < 1:
            raise ValueError(
                f"Prototype augmentation requires a positive batch size, got {sample_count}."
            )
        return sample_count


    def save_proto_aug_dict(self, save_path):
        proto_aug_dict = {
            'prototypes': self.prototypes,
            'radius': self.radius,
            'mean_similarity': self.mean_similarity,
        }

        torch.save(proto_aug_dict, save_path)

    # load continual   # NOTE!!!
    def load_proto_aug_dict(self, load_path):
        proto_aug_dict = torch.load(load_path)

        self.prototypes = proto_aug_dict['prototypes']
        self.radius = proto_aug_dict['radius']
        self.mean_similarity = proto_aug_dict['mean_similarity']


    def compute_proto_aug_loss(self, model, batch_size=None):
        sample_count = self._sample_count(batch_size)
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)
        prototypes_labels = torch.randint(0, len(prototypes), (sample_count,)).to(self.device)   # dtype=torch.long
        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((sample_count, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        #prototypes_augmented = F.normalize(prototypes_augmented, dim=-1, p=2) # NOTE!!! DO NOT normalize
        # forward prototypes and get logits
        _, prototypes_output = model[1](prototypes_augmented)
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss


    def compute_proto_aug_hardness_aware_loss(self, model, batch_size=None):
        sample_count = self._sample_count(batch_size)
        prototypes = F.normalize(self.prototypes, dim=-1, p=2).to(self.device)

        # hardness-aware sampling
        sampling_prob = F.softmax(self.mean_similarity / self.hardness_temp, dim=-1)
        sampling_prob = sampling_prob.cpu().numpy()
        prototypes_labels = np.random.choice(len(prototypes), size=(sample_count,), replace=True, p=sampling_prob)
        prototypes_labels = torch.from_numpy(prototypes_labels).long().to(self.device)

        prototypes_sampled = prototypes[prototypes_labels]
        prototypes_augmented = prototypes_sampled + torch.randn((sample_count, self.feature_dim), device=self.device) * self.radius * self.radius_scale
        #prototypes_augmented = F.normalize(prototypes_augmented, dim=-1, p=2) # NOTE!!! DO NOT normalize
        # forward prototypes and get logits
        _, prototypes_output = model[1](prototypes_augmented)
        proto_aug_loss = nn.CrossEntropyLoss()(prototypes_output / 0.1, prototypes_labels)

        return proto_aug_loss


    def update_prototypes_offline(self, model, train_loader, num_labeled_classes):
        model.eval()

        all_feats_list = []
        all_labels_list = []
        # forward data
        for batch_idx, (images, label, _) in enumerate(tqdm(train_loader)):   # NOTE!!!
            if isinstance(images, dict):
                images = move_trial_batch_to_device(images, self.device)
            else:
                images = images.to(self.device, non_blocking=True)
            with torch.no_grad():
                feats = model[0](images)   # backbone
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_labels_list.append(label.to(self.device, non_blocking=True))
        all_feats = torch.cat(all_feats_list, dim=0)
        all_labels = torch.cat(all_labels_list, dim=0)

        # compute prototypes and radius
        prototypes_list = []
        radius_list = []
        for c in range(num_labeled_classes):
            feats_c = all_feats[all_labels==c]
            feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)
            feats_c_center = feats_c - feats_c_mean
            cov = torch.matmul(feats_c_center.t(), feats_c_center) / len(feats_c_center)
            radius = torch.trace(cov) / self.feature_dim   # or feats_c_center.shape[1]
            radius_list.append(radius)
        avg_radius = torch.sqrt(torch.mean(torch.stack(radius_list)))
        prototypes_all = torch.stack(prototypes_list, dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # update
        self.radius = avg_radius
        self.prototypes = prototypes_all

        # update mean similarity for each prototype
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i,i] -= similarity[i,i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity)-1)

        self.mean_similarity = mean_similarity


    def update_prototypes_online(self, model, train_loader, num_seen_classes, num_all_classes):
        model.eval()

        all_preds_list = []
        all_feats_list = []
        # forward data
        for batch_idx, (images, label, _, _) in enumerate(tqdm(train_loader)):   # NOTE!!!
            if isinstance(images, dict):
                images = move_trial_batch_to_device(images, self.device)
            else:
                images = images.to(self.device, non_blocking=True)
            with torch.no_grad():
                _, logits = model(images)
                feats = model[0](images)   # backbone
                feats = torch.nn.functional.normalize(feats, dim=-1)
                all_feats_list.append(feats)
                all_preds_list.append(logits.argmax(1))
        all_feats = torch.cat(all_feats_list, dim=0)
        all_preds = torch.cat(all_preds_list, dim=0)

        # compute prototypes
        prototypes_list = []
        for c in range(num_seen_classes, num_all_classes):
            feats_c = all_feats[all_preds==c]
            if len(feats_c) == 0:
                self.logger.info('No pred of this class, using fc (last_layer) parameters...')
                feats_c_mean = model[1].last_layer.weight_v.data[c]
            else:
                self.logger.info('computing (predicted) class-wise mean...')
                feats_c_mean = torch.mean(feats_c, dim=0)
            prototypes_list.append(feats_c_mean)
        prototypes_cur = torch.stack(prototypes_list, dim=0)   # NOTE!!!
        prototypes_all = torch.cat([self.prototypes, prototypes_cur], dim=0)
        prototypes_all = F.normalize(prototypes_all, dim=-1, p=2)

        # update
        self.prototypes = prototypes_all

        # update mean similarity for each prototype
        similarity = prototypes_all @ prototypes_all.T
        for i in range(len(similarity)):
            similarity[i,i] -= similarity[i,i]
        mean_similarity = torch.sum(similarity, dim=-1) / (len(similarity)-1)

        self.mean_similarity = mean_similarity
