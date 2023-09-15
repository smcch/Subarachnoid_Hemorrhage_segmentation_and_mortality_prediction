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
# ==============================================================================#

import os
import ants
import argparse
import subprocess
import pandas as pd
from fsl.wrappers import fslmaths, bet
from fpdf import FPDF
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# AUCMEDI libraries
from aucmedi import *
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi.data_processing.subfunctions import *
from aucmedi.xai import xai_decoder
from aucmedi.ensemble import predict_augmenting
from aucmedi import ImageAugmentation, DataGenerator
from PIL import Image, ImageDraw
from PIL import ImageFont


def dicom_to_nifti(dicom_dir, output_dir):
    # Extract the directory name
    dicom_folder_name = os.path.basename(dicom_dir)

    # Set the output filename to be the name of the DICOM directory
    command = ['dcm2niix', '-f', dicom_folder_name, '-p', 'y', '-z', 'y', '-m', 'y', '-o', output_dir, dicom_dir]
    subprocess.run(command)


def brain(image):
    affine = image.affine
    header = image.header
    tmpfile = 'tmpfile.nii.gz'
    image.to_filename(tmpfile)

    # FSL calls
    mask = fslmaths(image).thr('0.000000').uthr('100.000000').bin().fillh().run()
    fslmaths(image).mas(mask).run(tmpfile)
    bet(tmpfile, tmpfile, fracintensity=0.01)
    mask = fslmaths(tmpfile).bin().fillh().run()
    image = fslmaths(image).mas(mask).run()
    image = nib.Nifti1Image(image.get_fdata(), affine, header)
    os.remove(tmpfile)

    return image


def nii2ants(image):
    ndim = image.ndim  # must be 3D
    q_form = image.get_qform()
    spacing = image.header["pixdim"][1: ndim + 1]
    origin = np.zeros((ndim))
    origin[:3] = q_form[:3, 3]
    direction = np.diag(np.ones(ndim))
    direction[:3, :3] = q_form[:3, :3] / spacing[:3]

    image = ants.from_numpy(
        data=image.get_fdata(),
        origin=origin.tolist(),
        spacing=spacing.tolist(),
        direction=direction)
    return image


def ants2nii(image):
    array_data = image.numpy()
    affine = np.hstack([image.direction * np.diag(image.spacing), np.array(image.origin).reshape(3, 1)])
    affine = np.vstack([affine, np.array([0, 0, 0, 1.])])
    nii_image = nib.Nifti1Image(array_data, affine)
    return nii_image


def rigid(fixed, moving):
    kwargs = {'-n': 'nearestNeighbor'}
    tx = ants.registration(fixed, moving, type_of_transform='SyN', mask=None, grad_step=0.2, flow_sigma=3,
                           total_sigma=0, aff_metric='mattes', aff_sampling=64, syn_metric='mattes', **kwargs)
    moving_reg = tx['warpedmovout']
    return moving_reg


def preprocess(input_path, output_path):
    TEMPLATE_PATH = 'ct_template2mni.nii.gz'  # Provide the template path

    for patient_folder in os.listdir(input_path):
        dicom_dir = os.path.join(input_path, patient_folder)  # Path to DICOM folder
        nifti_dir = os.path.join(dicom_dir, 'NIFTI')  # Path to output NIFTI files
        os.makedirs(nifti_dir, exist_ok=True)

        # Convert DICOM to NIfTI
        dicom_to_nifti(dicom_dir, nifti_dir)

        nifti_files = [file for file in os.listdir(nifti_dir) if file.endswith('.nii') or file.endswith('.nii.gz')]

        # If more than one NIfTI file, select only the one with 'Tilt' in the name
        if len(nifti_files) > 1:
            nifti_files = [file for file in nifti_files if 'Tilt' in file]

        for nifti_file in nifti_files:
            filename = os.path.join(nifti_dir, nifti_file)
            original_image = nib.load(filename)
            template = nib.load(TEMPLATE_PATH)
            bet_image = brain(original_image)
            image_ant = nii2ants(bet_image)
            fixed_ant = nii2ants(template)
            moving_reg = rigid(fixed_ant, image_ant)
            registered_nii = ants2nii(moving_reg)

            out_dir = os.path.join(output_path, f'output_{patient_folder}')  # Path to output folder
            os.makedirs(out_dir, exist_ok=True)
            output_filename = f"{patient_folder}_ct.nii.gz"
            nib.save(registered_nii, os.path.join(out_dir, output_filename))


def predict_aucmedi(path_images, path_model, path_output, path_xai=None, gpu=0):
    # Define some parameters
    batch_size = 8
    batch_queue_size = 10
    processes = 4
    threads = 4

    # Define architecture which should be processed
    architecture = "3D.DenseNet121"

    # Define input shape
    resampling = (1.10, 1.58, 1.18)
    input_shape = (160, 128, 128)

    # Set dynamic growth
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # Fix GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Pillar #1: Initialize input data reader
    ds = input_interface(interface="directory", path_imagedir=path_images, training=False)
    (index_list, _, _, _, image_format) = ds

    # Define Subfunctions
    sf_list = [Clip(min=0, max=100),
               Standardize(mode="grayscale"),
               Padding(mode="constant", shape=input_shape),
               Crop(shape=input_shape, mode="center"),
               Chromer(target="rgb")]

    # Pillar #2: Initialize model
    model = NeuralNetwork(n_labels=2, channels=3,
                          architecture=architecture,
                          input_shape=input_shape,
                          workers=processes,
                          batch_queue_size=batch_queue_size,
                          multiprocessing=False)
    # Load model
    model.load(path_model)

    # Pillar #3: Initialize testing datagenerator
    test_gen = DataGenerator(index_list, path_images,
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
                             resampling=resampling,
                             )

    # Compute predictions
    preds = model.predict(prediction_generator=test_gen)

    # Store predictions to disk
    df_index = pd.DataFrame(data={"index": index_list})
    df_pd = pd.DataFrame(data=preds, columns=["pd_ASH:0", "pd_ASH:1"])
    df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)

    # Save the predictions.csv file in the correct output directory
    output_dir = os.path.join(path_output, os.path.basename(path_images), 'aucmedi')
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    df_merged.to_csv(predictions_file, index=False)

    # Compute XAI if desired
    if path_xai is not None:
        xai_decoder(test_gen, model, preds, overlay=True, out_path=output_dir)

from fpdf import FPDF


def generate_report(output_dir, subject_id, volume_nifti, xai_nifti, probability, segmentation_nifti):
    # Create a PDF report
    report_file = os.path.join(output_dir, f'report_{subject_id}.pdf')
    c = FPDF()

    # Set up the PDF report
    c.set_auto_page_break(auto=True, margin=15)
    c.add_page()

    # Set the font for the report
    c.set_font("Arial", size=12)

    # Add the header image
    header_image = 'unvrh.png'  # Replace with the absolute path to the image file
    c.image(header_image, x=c.w - 60, y=10, w=50)

    # Add the subject ID and probability to the report
    c.set_font("Arial", size=16, style="B")
    c.ln(60)
    c.cell(0, 15, f"Subject ID: {subject_id}", ln=True, align='L')
    c.set_font("Arial", size=16)
    c.cell(0, 15, f"Probability of Death: {probability:.2f}%", ln=True, align='L')

    # Calculate the hemorrhage volume in cc³
    seg_img = nib.load(segmentation_nifti)
    seg_data = seg_img.get_fdata()
    hemorrhage_volume = np.sum(seg_data) / 1000.0  # Divide by 1000 and display with two decimals
    c.cell(0, 15, f"Hemorrhage Volume: {hemorrhage_volume:.2f} cm³", ln=True, align='L')

    # Generate volume slices
    volume_slices_file = os.path.join(output_dir, 'volume_slices.png')
    create_volume_grid(volume_nifti, segmentation_nifti, volume_slices_file)

    # Add volume slices to the PDF report
    c.set_font("Arial", size=14, style="B")
    c.ln(20)
    c.cell(0, 15, "Hemorrhage Segmentation", ln=True, align='L')
    c.image(volume_slices_file, x=10, y=c.get_y() + 10, w=180, h=120)
    c.ln(130)

    # Start a new page for XAI slices
    c.add_page()

    # Generate XAI slices
    xai_slices_file = os.path.join(output_dir, 'xai_slices.png')
    create_xai_grid(xai_nifti, xai_slices_file)

    # Add XAI slices to the PDF report
    c.set_font("Arial", size=14, style="B")
    c.cell(0, 15, "XAI Slices", ln=True, align='L')
    c.image(xai_slices_file, x=10, y=c.get_y() + 10, w=180, h=120)
    c.ln(130)

#    # Start a new page for Segmentation overlay
#    c.add_page()
#
#    # Generate segmentation overlay slices
#    seg_overlay_slices_file = os.path.join(output_dir, 'seg_overlay_slices.png')
#    create_volume_grid(volume_nifti, segmentation_nifti, seg_overlay_slices_file)
#
#    # Add Segmentation overlay slices to the PDF report
#    c.set_font("Arial", size=14, style="B")
#    c.cell(0, 15, "Segmentation Overlay Slices", ln=True, align='L')
#    c.image(seg_overlay_slices_file, x=10, y=c.get_y() + 10, w=180, h=120)
#    c.ln(130)

    # Set the footer style
    c.set_font("Arial", size=12, style="I")
    c.set_text_color(128)

    # Add the footer
    c.cell(0, 10, "N.I.C.A.R.U. - Neurosurgical Imaging and Computational Analysis Research Unit", 0, 0, 'C')
    c.ln()  # Move to the next line
    c.cell(0, 10, "Río Hortega University Hospital - Valladolid - Spain", 0, 0, 'C')

    # Save and close the PDF report
    c.output(report_file)

    return f"PDF report generated: {report_file}"


def create_volume_grid(volume_nifti, segmentation_nifti, output_file):
    import os
    import numpy as np
    import nibabel as nib
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    # Check if the segmentation_nifti file exists
    if not os.path.exists(segmentation_nifti):
        raise FileNotFoundError(f"No such file or no access: '{segmentation_nifti}'")

    # Load the NIfTI volume and segmentation
    volume_img = nib.load(volume_nifti)
    volume_data = volume_img.get_fdata()
    seg_img = nib.load(segmentation_nifti)
    seg_data = seg_img.get_fdata()

    # Exclude the top and bottom slices
    volume_data = volume_data[:, :, 1:-1]
    seg_data = seg_data[:, :, 1:-1]

    # Create a 4x4 grid for plotting
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    # Custom colormap for segmentation: everything transparent except for red
    colors = [(0, 0, 0, 0), (1, 0, 0)]  # R -> G -> B -> Alpha
    cmap_name = 'custom_div_cmap'
    seg_cmap = ListedColormap(colors, name=cmap_name)

    start_slice = 60
    increment = 3

    # Plot rotated axial slices with segmentation overlay
    for i in range(4):
        for j in range(4):
            slice_index = start_slice + (i * 4 + j) * increment

            if slice_index >= 0 and slice_index < volume_data.shape[2]:
                volume_slice = volume_data[:, :, slice_index]
                seg_slice = seg_data[:, :, slice_index]

                # Normalize the volume data to 0-1 range for proper color mapping
                volume_slice = (volume_slice - np.min(volume_slice)) / (np.max(volume_slice) - np.min(volume_slice))

                # Rotate the slices 90 degrees clockwise
                volume_slice = np.rot90(volume_slice, k=1)  # k=1 for 90 degrees anti clockwise
                seg_slice = np.rot90(seg_slice, k=1)

                # Display the rotated volume slice in gray
                axs[i, j].imshow(volume_slice, cmap='gray')

                # Overlay the rotated segmentation with the custom colormap
                axs[i, j].imshow(seg_slice, cmap=seg_cmap, alpha=0.5)

                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')

    # Save the grid of slices with overlay as a single image file
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)

    return f"Grid of slices with overlay saved as a single image: {output_file}"




def create_xai_grid(nifti_file, output_file):
    # Load the NIfTI volume
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Exclude the top and bottom slices
    data = data[:, :, 1:-1, 0]

    # Get the shape of the data
    num_slices = data.shape[2]

    # Get the indices for the middle slices
    middle_slice_index = num_slices // 2

    # Determine the starting and ending indices for the grid
    start_index = middle_slice_index - 7  # Show 4 slices before and 3 slices after the middle
    end_index = middle_slice_index + 4  # Show 4 slices after the middle

    # Create a 4x4 grid for plotting
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))

    start_slice = 60
    increment = 3

    # Iterate through the grid and plot the slices
    for i in range(4):
        for j in range(4):
            slice_index = start_slice + (i * 4 + j) * increment

            if slice_index >= 0 and slice_index < num_slices:
                slice_data = data[:, :, slice_index]

                # Normalize the data to 0-1 range for proper color mapping
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data))

                # Increase the contrast by adjusting the intensity range
                slice_data = np.clip(slice_data * 2.0 - 0.5, 0.0, 1.0)

                # Rotate the slice by 180 degrees
                rotated_slice = np.rot90(slice_data, k=1)

                axs[i, j].imshow(rotated_slice, cmap='gray', vmin=0.0, vmax=1.0)
                axs[i, j].axis('off')

            else:
                axs[i, j].axis('off')

    # Save the grid of slices as a single image file
    fig.tight_layout()
    fig.savefig(output_file)
    plt.close(fig)

    print(f"Grid of slices saved as a single image: {output_file}")


def main(input_path, output_path, model_path, gpu=0):
    preprocess(input_path, output_path)
    for patient_folder in os.listdir(output_path):
        patient_dir = os.path.join(output_path, patient_folder)
        predict_aucmedi(patient_dir, model_path, output_path, output_path, gpu)

        # Read the predictions.csv file
        predictions_file = os.path.join(patient_dir, 'aucmedi',
                                        'predictions.csv')  # CHANGE: Read from 'aucmedi' subfolder
        df_merged = pd.read_csv(predictions_file)

        # Extract the subject ID from the patient folder name
        subject_id = patient_folder.split("_")[-1]

        # Define the paths to the volume and XAI NIfTI files
        volume_nifti = os.path.join(patient_dir, f"{subject_id}_ct.nii.gz")
        xai_nifti = os.path.join(patient_dir, 'aucmedi',
                                 f"{subject_id}_ct.nii.gz")  # CHANGE: Read from 'aucmedi' subfolder

        # Generate the report
        probability = df_merged.iloc[0]['pd_ASH:1'] * 100
        generate_report(patient_dir, subject_id, volume_nifti, xai_nifti,
                        probability)  # CHANGE: Save in patient_dir instead of output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing and AUCMEDI prediction pipeline')
    parser.add_argument('-i', '--input', help='Input directory path to DICOM files', required=True)
    parser.add_argument('-o', '--output', help='Output directory path', required=True)
    parser.add_argument('--model', help='Path to the AUCMEDI fitted model', required=True)
    parser.add_argument('-g', '--gpu', help='GPU ID selection for multi cluster', required=False, type=int, default=0)
    args = parser.parse_args()

    main(args.input, args.output, args.model, args.gpu)
