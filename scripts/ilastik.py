import argparse
import glob
import os
import subprocess
import napari
import numpy as np
import h5py


def split_h5_file(file_to_split):
    outp_folder = os.path.splitext(file_to_split)[0]
    if not os.path.exists(outp_folder):
        os.mkdir(outp_folder)
    h5_file = h5py.File(file_to_split, 'r')
    for key in h5_file.keys():
        data = h5_file[key][:]
        if data.ndim == 3:
            data = data[np.newaxis, ...]
        new_h5 = h5py.File(os.path.join(outp_folder, key + '.h5'), 'w')
        _ = new_h5.create_dataset("raw", data=data, compression='gzip')


def run_ilastik(file_name, ilastik_exec, projects_folder):
    pixel_project = os.path.join(projects_folder, '/nuclei_pixel.ilp')
    class_project = os.path.join(projects_folder, 'nuclei_object.ilp')

    pixel_outname = os.path.splitext(file_name)[0] + '_Probabilities/' + '{nickname}'
    class_outname = os.path.splitext(file_name)[0] + '_Objects/' + '{nickname}'

    raw_files = glob.glob(os.path.splitext(file_name)[0] + '/*')
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


def view_cells(file_name):
    with h5py.File(file_name, 'r') as f:
        ids = list(f.keys())
    for i, cell_id in enumerate(ids):
        with h5py.File(os.path.splitext(file_name)[0] + '/' + cell_id + '.h5') as f:
            raw = f['raw'][0]
        with h5py.File(os.path.splitext(file_name)[0] + '_Objects/' + cell_id + '.h5') as f:
            obj = f['exported_data'][0, :, :, :, 0]
        viewer = napari.Viewer()
        viewer.add_image(raw, blending='additive')
        viewer.add_labels(obj, blending='additive')
        _ = input('{}, {}'.format(i, cell_id))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run ilastik segmentation')
    parser.add_argument('h5_file_name', type=str,
                        help='h5 file to run segmentation on')
    parser.add_argument('--ilastik_exec', type=str,
                        default='/home/zinchenk/software/ilastik-1.4.0b5-Linux/run_ilastik.sh',
                        help='path to ilastik executable')
    parser.add_argument('--projects_folder', type=str,
                        default='/home/zinchenk/work/brain_cells/ilastik/',
                        help='folder containing ilp projects')
    parser.add_argument('--view_cells', type=int, default=0, choices=[0, 1],
                        help='view resulting segmentaion in napari')
    args = parser.parse_args()

    split_h5_file(args.h5_file_name)
    run_ilastik(args.h5_file_name, args.ilastik_exec, args.projects_folder)
    if args.view_cells:
        view_cells(args.h5_file_name)
