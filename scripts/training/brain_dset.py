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
from skimage.measure import label


class CellDataset(Dataset):
    def __init__(self, volumes_file_name, labels_dset, segm_file_name=None,
                 transforms=None, train=False, ignore_labels=None, ae=False):
        self.volumes = z5py.File(volumes_file_name)
        self.segm = z5py.File(segm_file_name) if segm_file_name else None
        self.annot_cells = self.volumes[labels_dset][:]
        if ignore_labels:
            for i in ignore_labels:
                self.annot_cells = self.annot_cells[np.where(self.annot_cells[:, 1] != i)]
        # if we have binary classification, BCE loss might not like the target shape
        self.reshape_target = False if len(np.unique(self.annot_cells[:,1])) > 2 else True
        self.transforms = transforms
        self.ae = ae
        self.train = train
        # for validation we need to know which cells are on the edge to preferentially not choose them
        if not train:
            self.inner_dict = {key: [val1, val2] for key, val1, val2
                               in self.volumes['inner_assign'][:]}

    def get_weights(self):
        _, label_counts = np.unique(self.annot_cells[:,1], return_counts=True)
        label_weights = len(self.annot_cells) / label_counts
        sample_weights = label_weights[self.annot_cells[:,1]]
        return sample_weights

    def choose_best_volume(self, key, volumes):
        is_present_in = self.inner_dict[key]
        num_black_pixels = [np.sum(vol == 0) for vol in volumes]
        # good in both volumes - choose the one with better contrast
        if sum(is_present_in) == 2:
            better_vol_id = np.argmax(num_black_pixels)
        # good in only one - vhoose this volume
        elif sum(is_present_in) == 1:
            better_vol_id = np.argmax(is_present_in)
        #if on edge in both - choose the one with less background
        else:
            better_vol_id = np.argmax(num_black_pixels)
        return better_vol_id

    def __len__(self):
        return len(self.annot_cells)

    def __getitem__(self, idx):
        key, label = self.annot_cells[idx]
        if self.reshape_target:
            label = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        cell_volume = self.volumes[str(key)][:]
        if len(cell_volume) == 1:
            volume2choose = 0
        # if the cell is present in both xray volumes
        else:
            if self.train:
                # when training, load a random one
                volume2choose = np.random.randint(0, 2)
            else:
                # when predicting, choose the most complete
                volume2choose = self.choose_best_volume(key, cell_volume)
        cell_volume = cell_volume[volume2choose]
        if self.segm:
            segm = self.segm[str(key)][volume2choose] * 255
            cell_volume = np.stack([cell_volume, segm])
        if self.transforms:
            cell_volume = self.transforms(cell_volume)
        if not self.ae:
            return cell_volume, label
        else:
            return cell_volume, [label, cell_volume]


def get_transforms(transform_config):
    transforms = Compose()
    if transform_config.get('crop_pad_to_size'):
        crop_pad_to_size = transform_config.get('crop_pad_to_size')
        transforms.add(vol_transf.CropPad2Size(**crop_pad_to_size))
    if transform_config.get('random_crop'):
        random_crop = transform_config.get('random_crop')
        transforms.add(vol_transf.VolumeRandomCrop(**random_crop))
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
        transforms.add(gen_transf.Normalize())
    if transform_config.get('noise'):
        noise_config = transform_config.get('noise')
        transforms.add(vol_transf.AdditiveNoise(**noise_config))
    if transform_config.get('torch_batch'):
        transforms.add(gen_transf.AsTorchBatch(3))
    return transforms


def get_loaders(configuration_file, train=True):
    config = yaml2dict(configuration_file)
    file_name = config.get('file_name')
    segm_file = config.get('segm_file_name', None)

    tfs = [get_transforms(config.get(key))
                  for key in ['train_transforms', 'val_transforms']]
    dicts = ['train_dict', 'val_dict']
    is_train = [True, False]

    cell_dsets = [CellDataset(file_name, dset, segm_file_name=segm_file, transforms=tfs,
                              train=is_tr, **config.get('dataset_kwargs', {}))
                  for tfs, dset, is_tr in zip(tfs, dicts, is_train)]
    if train:
        samplers = [WeightedRandomSampler(dset.get_weights(), len(dset), replacement=True)
                    for dset in cell_dsets]
        train_loader = DataLoader(cell_dsets[0], sampler=samplers[0], **config.get('loader_config'))
        val_loader = DataLoader(cell_dsets[1], sampler=samplers[1], **config.get('val_loader_config'))
        return train_loader, val_loader
    else:
        val_loader = DataLoader(cell_dsets[1], **config.get('val_loader_config'))
        return val_loader
