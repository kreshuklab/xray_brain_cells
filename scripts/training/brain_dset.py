import h5py
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from inferno.io.transform import Compose
from inferno.io.transform import generic as gen_transf
from inferno.io.transform import volume as vol_transf
from inferno.io.transform.image import ElasticTransform
from inferno.utils.io_utils import yaml2dict


class CellDataset(Dataset):
    def __init__(self, volumes_file_name, labels_dset, transforms=None):
        self.volumes = h5py.File(volumes_file_name)
        self.annot_cells = self.volumes[labels_dset][:]
        self.transforms = transforms

    def __len__(self):
        return len(self.annot_cells)

    def __getitem__(self, idx):
        key, label = self.annot_cells[idx]
        cell_volume = self.volumes[str(key)][:]
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


def get_loaders(configuration_file):
    config = yaml2dict(configuration_file)
    transforms = get_transforms(config.get('transforms')) if config.get('transforms') else None
    file_name = config.get('file_name')

    cell_dsets = [CellDataset(file_name, dset, transforms=transforms)
                  for dset in ['train_dict', 'val_dict']]

    train_loader = DataLoader(cell_dsets[0], **config.get('loader_config'))
    val_loader = DataLoader(cell_dsets[1], **config.get('val_loader_config'))
    return train_loader, val_loader
