import os
import ants
import numpy as np
import nibabel as nib
import subprocess
import argparse
from fsl.wrappers import fslmaths, bet
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# AUCMEDI libraries
from aucmedi import *
from aucmedi.data_processing.io_loader import sitk_loader
from aucmedi.data_processing.subfunctions import *
from aucmedi.xai import xai_decoder
from aucmedi.ensemble import predict_augmenting
from aucmedi import ImageAugmentation, DataGenerator


def dicom_to_nifti(dicom_dir, output_dir):
    command = ['dcm2niix', '-o', output_dir, dicom_dir]
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
    TEMPLATE_PATH = '/mnt/c/Users/ncrhurh/PycharmProjects/HSA_CNN/ct_template2mni.nii.gz'  # Provide the template path

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
    output_dir = os.path.join(path_output, os.path.basename(path_images))
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = os.path.join(output_dir, 'predictions.csv')
    df_merged.to_csv(predictions_file, index=False)

    # Compute XAI if desired
    if path_xai is not None:
        xai_decoder(test_gen, model, preds, overlay=True, out_path=path_output)

import argparse
import subprocess
import pandas as pd
from fsl.wrappers import fslmaths, bet

from fpdf import FPDF
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def generate_report(output_dir, subject_id, volume_nifti, xai_nifti, probability):
    # Create a PDF report
    report_file = os.path.join(output_dir, f'report_{subject_id}.pdf')
    c = FPDF()

    # Set up the PDF report
    c.set_auto_page_break(auto=True, margin=15)
    c.add_page()

    # Set the font for the report
    c.set_font("Helvetica", size=12)

    # Add the subject ID and probability to the report
    c.set_font("Helvetica", size=16, style="B")
    c.cell(0, 15, f"Subject ID: {subject_id}", ln=True, align='L')
    c.cell(0, 15, f"Probability of Death: {probability:.2f}", ln=True, align='L')

    # Load the NIfTI volume
    volume_img = nib.load(volume_nifti)
    volume_data = volume_img.get_fdata()

    # Extract the center slices for axial, coronal, and sagittal views
    z_center = volume_data.shape[2] // 2
    axial_slice = volume_data[:, :, z_center].squeeze().transpose()
    coronal_slice = volume_data[:, z_center, :].squeeze().transpose()
    sagittal_slice = volume_data[z_center, :, :].squeeze().transpose()

    # Save the slices as a PNG file
    volume_slices_file = os.path.join(output_dir, 'volume_slices.png')
    plt.figure()
    plt.subplot(131)
    plt.imshow(axial_slice.T, cmap='gray')  # Transpose axial_slice
    plt.title('Axial')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(coronal_slice.T, cmap='gray')  # Transpose coronal_slice
    plt.title('Coronal')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(sagittal_slice.T, cmap='gray')  # Transpose sagittal_slice
    plt.title('Sagittal')
    plt.axis('off')

    plt.savefig(volume_slices_file, bbox_inches='tight')
    plt.close()

    # Draw the volume slices in the PDF report
    c.set_font("Helvetica", size=12)
    c.cell(0, 15, "Volume Slices:", ln=True, align='L')
    c.image(volume_slices_file, x=c.get_x(), y=c.get_y() + 10, w=180, h=120)
    c.ln(130)

    # Load the XAI NIfTI image
    xai_img = nib.load(xai_nifti)
    xai_data = xai_img.get_fdata()

    # Extract the center slices for axial, coronal, and sagittal views
    z_center = xai_data.shape[2] // 2
    axial_slice = xai_data[:, :, z_center].squeeze().transpose()
    coronal_slice = xai_data[:, z_center, :].squeeze().transpose()
    sagittal_slice = xai_data[z_center, :, :].squeeze().transpose()

    # Save the slices as a PNG file
    xai_slices_file = os.path.join(output_dir, 'xai_slices.png')
    plt.figure()
    plt.subplot(131)
    plt.imshow(axial_slice.T, cmap='gray')  # Transpose axial_slice
    plt.title('Axial')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(coronal_slice.T, cmap='gray')  # Transpose coronal_slice
    plt.title('Coronal')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(sagittal_slice.T, cmap='gray')  # Transpose sagittal_slice
    plt.title('Sagittal')
    plt.axis('off')

    plt.savefig(xai_slices_file, bbox_inches='tight')
    plt.close()

    # Draw the XAI slices in the PDF report
    c.set_font("Helvetica", size=12)
    c.cell(0, 15, "XAI Slices:", ln=True, align='L')
    c.image(xai_slices_file, x=c.get_x(), y=c.get_y() + 10, w=180, h=120)
    c.ln(130)

    # Save and close the PDF report
    c.output(report_file)



def main(input_path, output_path, model_path, gpu=0):
    preprocess(input_path, output_path)
    for patient_folder in os.listdir(output_path):
        patient_dir = os.path.join(output_path, patient_folder)
        predict_aucmedi(patient_dir, model_path, output_path, output_path, gpu)

        # Read the predictions.csv file
        predictions_file = os.path.join(patient_dir, 'predictions.csv')
        df_merged = pd.read_csv(predictions_file)

        # Extract the subject ID from the patient folder name
        subject_id = patient_folder.split("_")[-1]

        # Define the paths to the volume and XAI NIfTI files
        volume_nifti = os.path.join(patient_dir, f"{subject_id}_ct.nii.gz")
        xai_nifti = os.path.join(output_path, f"{subject_id}_ct.nii.gz")

        # Generate the report
        probability = df_merged.iloc[0]['pd_ASH:1'] * 100
        generate_report(output_path, subject_id, volume_nifti, xai_nifti, probability)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing and AUCMEDI prediction pipeline')
    parser.add_argument('-i', '--input', help='Input directory path to DICOM files', required=True)
    parser.add_argument('-o', '--output', help='Output directory path', required=True)
    parser.add_argument('--model', help='Path to the AUCMEDI fitted model', required=True)
    parser.add_argument('-g', '--gpu', help='GPU ID selection for multi cluster', required=False, type=int, default=0)
    args = parser.parse_args()

    main(args.input, args.output, args.model, args.gpu)
