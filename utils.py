# ==============================================================================#
#  Author:       Santiago Cepeda, Dominik Müller                              #
#  Copyright:    Río Hortega University Hospital in Valladolid, Spain          #
#                University of Augsburg, Germany                               #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
# =============================================================================#

import os
import subprocess
import shutil
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tensorflow as tf
import ants
import mri_synthstrip
from aucmedi import NeuralNetwork, DataGenerator, input_interface
from aucmedi.data_processing.subfunctions import Standardize, Padding, Crop, Chromer
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi.xai import xai_decoder
from matplotlib.colors import ListedColormap
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def run_nnunet_segmentation(input_folder, output_folder):
    """
    Performs segmentation using nnUNet.

    Args:
        input_folder (str): Path to the subject's folder containing the NIFTI file.
        output_folder (str): Path where the segmentation output will be saved.
    """
    nifti_dir = os.path.join(input_folder, 'NIFTI')
    if not os.path.exists(nifti_dir):
        logger.error(f"NIFTI folder not found in {input_folder}")
        return

    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')]
    if not nifti_files:
        logger.error(f"No .nii.gz files found in {nifti_dir}")
        return

    nifti_file = nifti_files[0]
    subject_id = os.path.basename(input_folder)
    formatted_input_dir = os.path.join(nifti_dir, subject_id)
    os.makedirs(formatted_input_dir, exist_ok=True)

    formatted_input_file = os.path.join(formatted_input_dir, f"{subject_id}_0000.nii.gz")
    try:
        shutil.copy(os.path.join(nifti_dir, nifti_file), formatted_input_file)
        logger.info(f"File copied and renamed: {formatted_input_file}")
    except Exception as e:
        logger.error(f"Error copying or renaming NIFTI file: {e}")
        return

    output_folder_2d = os.path.join(output_folder, "output_model_2d")
    output_folder_3d = os.path.join(output_folder, "output_model_3d_lowres")
    ensemble_output_folder = os.path.join(output_folder, "ensemble_output")

    os.makedirs(output_folder_2d, exist_ok=True)
    os.makedirs(output_folder_3d, exist_ok=True)
    os.makedirs(ensemble_output_folder, exist_ok=True)

    try:
        logger.info("Running segmentation script...")
        subprocess.run([
            "python", "bleeding_segmentation.py",
            "-i", formatted_input_dir,
            "-o", output_folder
        ], check=True)

        output_file = os.path.join(ensemble_output_folder, f"{subject_id}.nii.gz")
        preferred_output_file = os.path.join(ensemble_output_folder, f"{subject_id}_segmentation.nii.gz")

        if os.path.exists(output_file):
            os.rename(output_file, preferred_output_file)
            logger.info(f"Segmentation file renamed to: {preferred_output_file}")
        else:
            logger.error(f"Segmentation file {preferred_output_file} was not generated correctly.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Segmentation error: {e}")

def dicom_to_nifti(dicom_dir, output_dir):
    dicom_folder_name = os.path.basename(dicom_dir)
    command = ['dcm2niix', '-f', dicom_folder_name, '-p', 'y', '-z', 'y', '-m', 'y', '-o', output_dir, dicom_dir]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def clip_intensities(nifti_file, output_dir):
    img = nib.load(nifti_file)
    data = img.get_fdata()
    clipped_data = np.clip(data, 0, 100)
    clipped_filename = os.path.join(output_dir, os.path.basename(nifti_file).replace(".nii.gz", "_clipped.nii.gz"))
    clipped_img = nib.Nifti1Image(clipped_data, img.affine, header=img.header)
    nib.save(clipped_img, clipped_filename)
    return clipped_filename

def register_with_antspy(fixed_image_path, moving_image_path, output_path):
    fixed_image = ants.image_read(fixed_image_path)
    moving_image = ants.image_read(moving_image_path)
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='SyN')
    ants.image_write(registration['warpedmovout'], output_path)
    return output_path

def skull_strip_image(nifti_file, output_dir, model_path):
    skull_stripped_file = os.path.join(output_dir, os.path.basename(nifti_file).replace(".nii.gz", "_sk.nii.gz"))
    mask_filename = skull_stripped_file.replace("_sk.nii.gz", "_mask.nii.gz")
    mri_synthstrip.run(image=nifti_file, out=skull_stripped_file, mask=mask_filename, modelPath=model_path)
    return skull_stripped_file

def preprocess(input_path, output_path, template_path):
    nifti_dir = os.path.join(output_path, 'NIFTI')
    os.makedirs(nifti_dir, exist_ok=True)

    dicom_to_nifti(input_path, nifti_dir)

    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'synthstrip_models')
    atlas_image_with_skull = template_path
    final_processed_file = None
    nifti_files = [file for file in os.listdir(nifti_dir) if file.endswith('.nii.gz')]

    if len(nifti_files) > 1:
        nifti_files = [file for file in nifti_files if 'Tilt' in file]

    for file_name in nifti_files:
        logger.info(f"Processing {file_name}...")
        nifti_file = os.path.join(nifti_dir, file_name)

        clipped_file = clip_intensities(nifti_file, output_path)
        registered_file = os.path.join(output_path, os.path.basename(clipped_file).replace(".nii.gz", "_reg.nii.gz"))
        registered_file = register_with_antspy(atlas_image_with_skull, clipped_file, registered_file)
        skull_stripped_file = skull_strip_image(registered_file, output_path, model_path)

        final_processed_file = skull_stripped_file
        logger.info(f"Processing of {file_name} completed. Results saved in {output_path}")

    if final_processed_file:
        standard_name = os.path.join(output_path, f"{os.path.basename(output_path)}_processed.nii.gz")
        os.rename(final_processed_file, standard_name)
    return standard_name

def predict_aucmedi(nifti_file, path_model, path_output, path_xai=None, gpu=0):
    batch_size = 1
    batch_queue_size = 10
    processes = 4
    threads = 4
    architecture = "3D.DenseNet121"
    resampling = (1.10, 1.58, 1.18)
    input_shape = (160, 128, 128)

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    subject_id = os.path.basename(os.path.dirname(nifti_file))
    input_dir = os.path.join(path_output, 'input_nifti')
    os.makedirs(input_dir, exist_ok=True)
    input_nifti_file = os.path.join(input_dir, os.path.basename(nifti_file))
    if not os.path.exists(input_nifti_file):
        shutil.copy(nifti_file, input_nifti_file)

    ds = input_interface(interface="directory", path_imagedir=input_dir, training=False)
    (index_list, _, _, _, image_format) = ds

    sf_list = [Standardize(mode="grayscale"),
               Padding(mode="constant", shape=input_shape),
               Crop(shape=input_shape, mode="center"),
               Chromer(target="rgb")]

    model = NeuralNetwork(n_labels=2, channels=3,
                          architecture=architecture,
                          input_shape=input_shape,
                          workers=processes,
                          batch_queue_size=batch_queue_size,
                          multiprocessing=False)
    model.load(path_model)

    test_gen = DataGenerator(index_list, input_dir,
                             labels=None,
                             batch_size=batch_size,
                             data_aug=None,
                             shuffle=False,
                             subfunctions=sf_list,
                             resize=None,
                             standardize_mode=model.meta_standardize,
                             grayscale=True,
                             prepare_images=False,
                             sample_weights=None,
                             seed=0,
                             image_format=image_format,
                             workers=threads,
                             loader=sitk_loader,
                             resampling=resampling)

    preds = model.predict(prediction_generator=test_gen)
    tf.keras.backend.clear_session()

    output_dir = os.path.join(path_output, 'aucmedi')
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, f'{subject_id}_predictions.csv')
    pd.DataFrame(data=preds, columns=["pd_ASH:0", "pd_ASH:1"]).to_csv(predictions_file, index=False)

    xai_output_dir = os.path.join(output_dir, 'xai_overlay')
    os.makedirs(xai_output_dir, exist_ok=True)

    if path_xai is not None:
        with tf.device('/CPU:0'):
            xai_decoder(test_gen, model, preds, overlay=True, out_path=xai_output_dir)

    return predictions_file, xai_output_dir

def create_volume_grid(volume_nifti, segmentation_nifti, output_file):
    """
    Creates a volume grid with segmentation overlay.

    Args:
        volume_nifti (str): Path to the input NIfTI file.
        segmentation_nifti (str): Path to the segmentation NIfTI file.
        output_file (str): Path where the generated grid image will be saved.
    """
    volume_img = nib.load(volume_nifti)
    volume_data = np.clip(volume_img.get_fdata(), 0, 100)
    seg_img = nib.load(segmentation_nifti)
    seg_data = seg_img.get_fdata()

    volume_data = volume_data[:, :, 10:-10]
    seg_data = seg_data[:, :, 10:-10]
    total_slices = volume_data.shape[2]
    selected_slices = np.linspace(5, total_slices - 5, 16, dtype=int)

    fig, axs = plt.subplots(4, 4, figsize=(15, 15))
    seg_cmap = ListedColormap([(0, 0, 0, 0), (1, 0, 0, 0.5)], name='custom_div_cmap')

    for i, ax in enumerate(axs.flat):
        slice_index = selected_slices[i]
        volume_slice = (volume_data[:, :, slice_index] - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data))
        seg_slice = seg_data[:, :, slice_index]

        ax.imshow(np.rot90(volume_slice), cmap='gray')
        ax.imshow(np.rot90(seg_slice), cmap=seg_cmap, alpha=0.5)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    logger.info(f"Grid of slices with overlay saved: {output_file}")

def create_xai_grid(xai_file, subject_id, output_file):
    """
    Creates an XAI grid of RGB slices.

    Args:
        xai_file (str): Path to the XAI-generated NIfTI file.
        subject_id (str): Subject/patient ID.
        output_file (str): Path where the generated grid image will be saved.
    """
    if not os.path.exists(xai_file):
        raise FileNotFoundError(f"XAI file not found: {xai_file}")

    img = nib.load(xai_file)
    data = np.clip(np.squeeze(img.get_fdata(), axis=3), 0, 255).astype(np.uint8)

    data_rgb = data[:, :, 10:-10, :]
    selected_slices = np.linspace(5, data_rgb.shape[2] - 5, 16, dtype=int)

    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    for i, ax in enumerate(axs.flat):
        slice_data = np.flipud(np.rot90(data_rgb[:, :, selected_slices[i], :], k=3))
        ax.imshow(slice_data)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    logger.info(f"Grid of RGB XAI slices saved: {output_file}")

def generate_report(output_dir, subject_id, volume_nifti, xai_nifti, probability, segmentation_nifti):
    """
    Generates a PDF report with volumes, segmentations, and XAI maps.

    Args:
        output_dir (str): Output directory where the report will be saved.
        subject_id (str): Subject/patient ID.
        volume_nifti (str): Path to the original NIfTI volume file.
        xai_nifti (str): Path to the XAI generated NIfTI file.
        probability (float): Probability calculated by the model.
        segmentation_nifti (str): Path to the segmentation NIfTI file.
    """
    report_file = os.path.join(output_dir, f'report_{subject_id}.pdf')
    c = FPDF()
    c.set_auto_page_break(auto=True, margin=15)
    c.add_page()
    c.set_font("Arial", size=12)

    header_image = 'unvrh.png'
    c.image(header_image, x=c.w - 60, y=10, w=50)

    c.set_font("Arial", size=16, style="B")
    c.ln(60)
    c.cell(0, 15, f"Subject ID: {subject_id}", ln=True, align='L')
    c.set_font("Arial", size=16)
    c.cell(0, 15, f"Probability of Death: {probability:.2f}%", ln=True, align='L')

    seg_img = nib.load(segmentation_nifti)
    seg_data = seg_img.get_fdata()
    voxel_volume = np.prod(seg_img.header.get_zooms())  # Get voxel size (x, y, z)
    hemorrhage_volume = np.sum(seg_data > 0) * voxel_volume / 1000.0  # Convert to cubic centimeters (cm³)
    c.cell(0, 15, f"Hemorrhage Volume: {hemorrhage_volume:.2f} cm³", ln=True, align='L')

    volume_slices_file = os.path.join(output_dir, 'volume_slices.png')
    create_volume_grid(volume_nifti, segmentation_nifti, volume_slices_file)
    c.set_font("Arial", size=14, style="B")
    c.ln(20)
    c.cell(0, 15, "Hemorrhage Segmentation", ln=True, align='L')
    c.image(volume_slices_file, x=10, y=c.get_y() + 10, w=180, h=120)
    c.ln(130)

    c.add_page()
    xai_slices_file = os.path.join(output_dir, 'xai_slices.png')
    create_xai_grid(xai_nifti, subject_id, xai_slices_file)
    c.set_font("Arial", size=14, style="B")
    c.cell(0, 15, "XAI Slices", ln=True, align='L')
    c.image(xai_slices_file, x=10, y=c.get_y() + 10, w=180, h=120)
    c.ln(130)

    c.set_font("Arial", size=12, style="I")
    c.set_text_color(128)
    c.cell(0, 10, "G.E.I.B.A.C. - Grupo Especializado en Imagen Biomédica y Análisis Computacional", 0, 0, 'C')
    c.ln()
    c.cell(0, 10, "Río Hortega University Hospital - Valladolid - Spain", 0, 0, 'C')
    c.output(report_file)

    print(f"PDF report generated: {report_file}")
    return hemorrhage_volume  # Return the calculated volume
