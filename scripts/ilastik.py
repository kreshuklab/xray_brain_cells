import argparse
import glob
import os
import shutil
import subprocess
import napari
import numpy as np
import h5py
import z5py
from scipy.ndimage import morphology
from skimage import measure


def get_keys(file_to_split):
    with z5py.File(file_to_split, 'r') as f:
        ids = list(f.keys())
    keys = [idx for idx in ids if idx.isdigit()]
    return keys


def split_n5_file(file_to_split, keys):
    outp_folder = os.path.splitext(file_to_split)[0]
    if not os.path.exists(outp_folder):
        os.mkdir(outp_folder)
    n5_file = z5py.File(file_to_split)
    for key in keys:
        data = n5_file[key][:]
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        new_h5 = h5py.File(os.path.join(outp_folder, key + '.h5'), 'w')
        _ = new_h5.create_dataset("raw", data=data, compression='gzip')


def run_ilastik(path, ilastik_exec, projects_folder):
    pixel_project = os.path.join(projects_folder, 'nuclei_pixel.ilp')
    class_project = os.path.join(projects_folder, 'nuclei_object.ilp')

    pixel_outname = path + '_Probabilities/' + '{nickname}'
    class_outname = path + '_Objects/' + '{nickname}'

    raw_files = glob.glob(path + '/*')
    prob_files = ['{}_Probabilities/{}'.format(*os.path.split(path)) for path in raw_files]

    pix_cmd = [ilastik_exec, '--headless', '--project=' + pixel_project,
               '--output_format=compressed hdf5', '--output_filename_format=' + pixel_outname,
               '--export_source=probabilities', *raw_files]
    obj_cmd = [ilastik_exec, '--headless', '--project=' + class_project,
               '--output_format=compressed hdf5', '--output_filename_format=' + class_outname,
               '--export_source=Object Predictions', '--raw_data', *raw_files,
               '--prediction_maps', *prob_files]

    subprocess.run(pix_cmd)
    subprocess.run(obj_cmd)


def postprocess(volume, key, erode=5, dilate=15):
    eroded = morphology.binary_erosion(volume, iterations=erode)
    cc = measure.label(eroded, connectivity=2)
    labels, counts = np.unique(cc, return_counts=True)
    if len(labels) == 1:
        return np.ones_like(volume)
    labels = labels[np.where(counts > 5000)]
    counts = counts[np.where(counts > 5000)]
    biggest_label = labels[1:][np.argmax(counts[1:])]
    distances = [np.min(DIST_CENTER[cc == label]) for label in labels]
    central_label = labels[1:][np.argmin(distances[1:])]
    if biggest_label != central_label:
        print('WARNING: in {} biggest_label != central_label'.format(key))
    needed_cell = cc == central_label
    dilated = morphology.binary_dilation(needed_cell, iterations=dilate)
    return dilated


def postprocess_and_merge(path, keys):
    out_f_name = path + '_segmented.n5'
    out_f = z5py.File(out_f_name, 'w')
    for cell_id in keys:
        with h5py.File(path + '_Objects/' + cell_id + '.h5') as f:
            segm = f['exported_data'][:, :, :, :, 0]
        processed = np.stack([postprocess(s, cell_id) for s in segm]).astype(int)
        _ = out_f.create_dataset(cell_id, data=processed, compression='gzip')
    out_f.close()
    shutil.rmtree(path)
    shutil.rmtree(path + '_Probabilities/')
    shutil.rmtree(path + '_Objects/')


def view_cell(idx, path):
    with h5py.File(path + '/' + idx + '.h5') as f:
        raw = f['raw'][0]
    with h5py.File(path + '_Objects/' + idx + '.h5') as f:
        segm = f['exported_data'][0, :, :, :, 0]
    viewer = napari.Viewer()
    viewer.add_image(raw, blending='additive')
    viewer.add_labels(segm, blending='additive')
    viewer.add_labels(postprocess(segm), blending='additive')


def view_all(path, keys):
    for i, cell_id in enumerate(keys):
        view_cell(cell_id, path)
        _ = input('{}, {}'.format(i, cell_id))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run ilastik segmentation')
    parser.add_argument('n5_file_name', type=str,
                        help='n5 file to run segmentation on')
    parser.add_argument('--ilastik_exec', type=str,
                        default='/home/zinchenk/software/ilastik-1.4.0b5-Linux/run_ilastik.sh',
                        help='path to ilastik executable')
    parser.add_argument('--projects_folder', type=str,
                        default='/home/zinchenk/work/brain_cells/ilastik/',
                        help='folder containing ilp projects')
    parser.add_argument('--only_view_cells', type=int, default=0, choices=[0, 1],
                        help='just view resulting segmentation in napari')
    args = parser.parse_args()

    file_prefix = os.path.splitext(args.n5_file_name)[0]
    cell_ids = get_keys(args.n5_file_name)
    split_n5_file(args.n5_file_name, cell_ids)
    run_ilastik(file_prefix, args.ilastik_exec, args.projects_folder)

    center_volume = np.ones((160, 160, 160))
    center_volume[79:81, 79:81, 79:81] = 0
    DIST_CENTER = morphology.distance_transform_edt(center_volume)

    if args.only_view_cells:
        view_all(file_prefix, cell_ids)
    else:
        postprocess_and_merge(file_prefix, cell_ids)
