import sys
import z5py
import numpy as np
import pandas as pd
import pymaid
from pymaid import tiles


def parse_annotations():
    # fetch all the neurons with given annotations
    all_neurons = [pymaid.find_neurons(annotations=i) for i in TYPES]
    # a list of skeleton_ids available
    skeleton_ids = [sk_id for neuron_type in all_neurons
                    for sk_id in neuron_type.skeleton_id.astype(int)]
    # dictionary skeleton_id : class_label
    label_dict = {i: label for label, neuron_type in enumerate(all_neurons)
                  for i in neuron_type.skeleton_id.astype(int)}
    pymaid.clear_cache()

    nodes_list = []
    no_soma_cells = []
    for neuron_type in all_neurons:
        nodes = neuron_type.nodes
        somas = neuron_type.soma
        # extract only the soma nodes
        nodes_list.append(nodes[nodes['treenode_id'].isin(somas)])
        # some cells might be missing somas, we won't use them
        if None in somas:
            no_soma_cells.append(neuron_type[neuron_type.soma == None].skeleton_id.astype(int))

    soma_coords = pd.concat(nodes_list)[['x', 'y', 'z']]

    for cell in np.array(no_soma_cells).flatten():
        skeleton_ids.remove(cell)
        del label_dict[cell]

    pymaid.clear_cache()
    return soma_coords, np.array(skeleton_ids), label_dict


def assign_to_volumes(center_coords, ids):
    # figure out which volume evry cell is present in
    is_in_volume1 = pymaid.in_volume(center_coords, 'r1_boundary')
    is_in_volume2 = pymaid.in_volume(center_coords, 'r2_boundary')
    pymaid.clear_cache()

    volume_assign = {}
    volume_assign.update({idx: 'volume1' for idx in ids[np.where(is_in_volume1 & ~is_in_volume2)]})
    volume_assign.update({idx: 'volume2' for idx in ids[np.where(~is_in_volume1 & is_in_volume2)]})
    volume_assign.update({idx: 'absent' for idx in ids[np.where(~is_in_volume1 & ~is_in_volume2)]})
    volume_assign.update({idx: 'both' for idx in ids[np.where(is_in_volume1 & is_in_volume2)]})

    return volume_assign


def fetch_volume(coords, stack, radius=8000):
    offs_coords = coords - VOLUME_OFFSETS[stack]
    min_max = [offs_coords - radius, offs_coords + radius]
    bbox = [c for co in zip(*min_max) for c in co]
    job = tiles.TileLoader(bbox, stack_id=STACK_IDS[stack], coords='NM')
    job.load_in_memory()
    return job.img


def is_on_border(vol, thres=0.02):
    # too many black pixels show that the cell is on a volume border
    total_volume = np.prod(vol.shape)
    return (np.sum(vol == 0) / total_volume) >= thres


def split_2_train_val(class_dict, split=0.2, seed=73):
    classes_array = np.array(list(class_dict.items()))
    np.random.seed(seed=seed)
    np.random.shuffle(classes_array)
    class_types = np.unique(classes_array[:, 1])
    classes_sorted = [classes_array[np.where(classes_array[:, 1] == type_)]
                    for type_ in class_types]
    classes_splits = [int(np.floor(len(one_class) * split))
                    for one_class in classes_sorted]
    training_set = np.vstack([one_class[class_split:] for one_class, class_split
                               in zip(classes_sorted, classes_splits)])
    validation_set = np.vstack([one_class[:class_split] for one_class, class_split
                               in zip(classes_sorted, classes_splits)])
    return training_set, validation_set


def save_volumes(soma_coords, skeleton_ids, label_dict, output_file_name):
    volume_assign = assign_to_volumes(soma_coords, skeleton_ids)
    output_file = z5py.File(output_file_name, 'w')

    for n, idx in enumerate(set(skeleton_ids)):
        if n < 310: continue
        print("Processed {} cells".format(n))
        center = soma_coords.iloc[np.where(skeleton_ids == idx)[0][0]].values.copy()
        volume = volume_assign[idx]

        if volume in ('volume1', 'volume2'):
            vol = fetch_volume(center, volume)
            volume2save = [vol] if ~is_on_border(vol) else None
        elif volume == 'both':
            vols = [fetch_volume(center, vol) for vol in ('volume1', 'volume2')]
            volume2save = [vol for vol in vols if ~is_on_border(vol)]
        else:
            volume2save = None

        pymaid.clear_cache()

        if volume2save:
            volume2save = np.squeeze(np.stack([volume2save])).astype('uint8')
            _ = output_file.create_dataset(str(idx), data=volume2save)
        else:
            del label_dict[idx]

    train_labels, val_labels = split_2_train_val(label_dict)
    _ = output_file.create_dataset('train_dict', data=train_labels)
    _ = output_file.create_dataset('val_dict', data=val_labels)


if __name__ == "__main__":

    outp_file = sys.argv[1]

    _ = pymaid.CatmaidInstance('http://catmaid3.hms.harvard.edu/catmaidppc/',
                                'kreshuk_lab', 'asdfjkl;',
                                '249b9498c98e5f9475f33f7084035ad191db24bf',
                                project_id=8)

    TYPES = ['pyramidal', 'not pyramidal', 'non neuronal\?', 'unclassified']
    STACK_IDS = {'volume1': 23, 'volume2': 27}
    VOLUME_OFFSETS = {'volume1': [-58400, -18800, -93600],
                      'volume2': [-146150, -27200, 40800]}

    soma_centers, cell_ids, class_labels = parse_annotations()
    save_volumes(soma_centers, cell_ids, class_labels, outp_file)
