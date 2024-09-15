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
import argparse
import gc
import pandas as pd
import tensorflow as tf
from utils import preprocess, run_nnunet_segmentation, predict_aucmedi, generate_report
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()


def main(input_path, output_path, gpu=0):
    # Define paths for the template and model files
    template_path = os.path.join(os.path.dirname(__file__), 'template.nii.gz')
    model_path = os.path.join(os.path.dirname(__file__), '3D.DenseNet121.model.best.loss.hdf5')

    # Iterate over each patient folder in the input directory
    for patient_folder in os.listdir(input_path):
        patient_input_dir = os.path.join(input_path, patient_folder)
        patient_output_dir = os.path.join(output_path, patient_folder)
        os.makedirs(patient_output_dir, exist_ok=True)

        # Preprocess the patient data
        print(f"Processing Subject ID: {patient_folder} - Preprocessing...")
        processed_file = preprocess(patient_input_dir, patient_output_dir, template_path)

        # Check if preprocessing was successful
        if os.path.exists(processed_file):
            segmentation_output_dir = os.path.join(patient_output_dir, 'segmentations')
            print(f"Processing Subject ID: {patient_folder} - Segmentation...")
            run_nnunet_segmentation(patient_output_dir, segmentation_output_dir)

            subject_id = patient_folder
            input_nifti = os.path.join(patient_output_dir, 'NIFTI', subject_id, f"{subject_id}_0000.nii.gz")
            segmentation_nifti = os.path.join(segmentation_output_dir, 'ensemble_output', f"{subject_id}_segmentation.nii.gz")

            # Check if the input NIfTI file exists
            if not os.path.exists(input_nifti):
                print(f"Error: Input NIfTI file not found: {input_nifti}")
                continue

            # Check if the segmentation output was generated
            if not os.path.exists(segmentation_nifti):
                print(f"Error: Segmentation file not generated correctly: {segmentation_nifti}")
                continue

            # Run AUCMEDI prediction and XAI generation
            print(f"Processing Subject ID: {patient_folder} - AUCMEDI Prediction...")
            predictions_file, xai_output_dir = predict_aucmedi(processed_file, model_path, patient_output_dir, patient_output_dir, gpu)

            # Load the predictions and extract the probability of interest
            df_merged = pd.read_csv(predictions_file)
            probability = df_merged.iloc[0]['pd_ASH:1'] * 100

            # Adjust the path to the XAI NIfTI file
            xai_nifti = os.path.join(xai_output_dir, f'{subject_id}_processed.nii.gz')

            # Check if the XAI output was generated correctly
            if not os.path.exists(xai_nifti):
                print(f"Error: XAI file not generated correctly: {xai_nifti}")
                continue

            # Generate the final report and get the hemorrhage volume
            print(f"Processing Subject ID: {patient_folder} - Generating Report...")
            hemorrhage_volume = generate_report(
                output_dir=patient_output_dir,
                subject_id=subject_id,
                volume_nifti=input_nifti,
                xai_nifti=xai_nifti,
                probability=probability,
                segmentation_nifti=segmentation_nifti
            )

            # Update predictions CSV file with the volume
            predictions_df = pd.read_csv(predictions_file)
            predictions_df['vol'] = hemorrhage_volume  # Add the volume column
            predictions_df.to_csv(predictions_file, index=False)
            print(f"Updated predictions file with volume: {predictions_file}")

            # Clear TensorFlow session and collect garbage to manage GPU memory
            gc.collect()
            tf.keras.backend.clear_session()

        print(f"Subject ID: {patient_folder} - Processing Completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing and AUCMEDI prediction pipeline')
    parser.add_argument('-i', '--input', help='Input directory path to DICOM files', required=True)
    parser.add_argument('-o', '--output', help='Output directory path', required=True)
    parser.add_argument('-g', '--gpu', help='GPU ID selection for multi cluster', required=False, type=int, default=0)
    args = parser.parse_args()

    main(args.input, args.output, args.gpu)