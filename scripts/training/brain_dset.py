import numpy as np
import z5py
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from inferno.io.transform import Compose
from inferno.io.transform import generic as gen_transf
from inferno.io.transform import volume as vol_transf
from inferno.io.transform.image import ElasticTransform
from inferno.utils.io_utils import yaml2dict


class CellDataset(Dataset):
    def __init__(self, volumes_file_name, labels_dset, transforms=None, ignore_labels=None):
        self.volumes = z5py.File(volumes_file_name)
        self.annot_cells = self.volumes[labels_dset][:]
        if ignore_labels:
            for i in ignore_labels:
                self.annot_cells = self.annot_cells[np.where(self.annot_cells[:, 1] != i)]
        # if we have binary classification, BCE loss might not like the target shape
        self.reshape_target = False if len(np.unique(self.annot_cells[:,1])) > 2 else True
        self.transforms = transforms

    def get_weights(self):
        _, label_counts = np.unique(self.annot_cells[:,1], return_counts=True)
        label_weights = len(self.annot_cells) / label_counts
        sample_weights = label_weights[self.annot_cells[:,1]]
        return sample_weights

    def __len__(self):
        return len(self.annot_cells)

    def __getitem__(self, idx):
        key, label = self.annot_cells[idx]
        if self.reshape_target:
            label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        cell_volume = self.volumes[str(key)][:]
        # if the cell is present in both xray volumes load random one
        if cell_volume.ndim == 4 and len(cell_volume) == 2:
            volume2choose = np.random.randint(0, 2)
            cell_volume = cell_volume[volume2choose]
        if self.transforms:
            cell_volume = self.transforms(cell_volume)
        return cell_volume, label


def get_transforms(transform_config):
    transforms = Compose()
    if transform_config.get('crop_pad_to_size'):
        crop_pad_to_size = transform_config.get('crop_pad_to_size')
        transforms.add(vol_transf.CropPad2Size(**crop_pad_to_size))
    if transform_config.get('cast'):
        transforms.add(gen_transf.Cast('float32'))
    if transform_config.get('normalize_range'):
        normalize_range_config = transform_config.get('normalize_range')
        transforms.add(gen_transf.NormalizeRange(**normalize_range_config))
    if transform_config.get('flip'):
        transforms.add(vol_transf.RandomFlip3D())
    if transform_config.get('rotate'):
        rotate_config = transform_config.get('rotate')
        transforms.add(vol_transf.RandomRot3D(order=3, **rotate_config))
    if transform_config.get('elastic_transform'):
        elastic_config = transform_config.get('elastic_transform')
        transforms.add(ElasticTransform(order=3, **elastic_config))
    if transform_config.get('normalize'):
        normalize_config = transform_config.get('normalize')
        transforms.add(gen_transf.Normalize(**normalize_config))
    if transform_config.get('noise'):
        noise_config = transform_config.get('noise')
        transforms.add(vol_transf.AdditiveNoise(**noise_config))
    if transform_config.get('torch_batch'):
        transforms.add(gen_transf.AsTorchBatch(3))
    return transforms


def get_loaders(configuration_file, train=True):
    config = yaml2dict(configuration_file)
    tfs = [get_transforms(config.get(key))
                  for key in ['train_transforms', 'val_transforms']]
    ignore_classes = config.get('ignore_classes', None)
    file_name = config.get('file_name')

    cell_dsets = [CellDataset(file_name, dset, transforms=tfs[i], ignore_labels=ignore_classes)
                  for i, dset in enumerate(['train_dict', 'val_dict'])]
    if train:
        samplers = [WeightedRandomSampler(dset.get_weights(), len(dset), replacement=True)
                    for dset in cell_dsets]
        train_loader = DataLoader(cell_dsets[0], sampler=samplers[0], **config.get('loader_config'))
        val_loader = DataLoader(cell_dsets[1], sampler=samplers[1], **config.get('val_loader_config'))
        return train_loader, val_loader
    else:
        val_loader = DataLoader(cell_dsets[1], **config.get('val_loader_config'))
        return val_loader
