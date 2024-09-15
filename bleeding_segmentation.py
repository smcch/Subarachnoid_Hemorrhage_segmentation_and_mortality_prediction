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
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def configure_environment():
    """Configure nnUNet environment variables."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    nnUNet_raw = os.path.join(dir_path, "my_nnunet", "nnUNet_raw")
    nnUNet_preprocessed = os.path.join(dir_path, "my_nnunet", "nnUNet_preprocessed")
    nnUNet_results = os.path.join(dir_path, "my_nnunet", "nnUNet_results")

    os.environ['nnUNet_raw'] = nnUNet_raw
    os.environ['nnUNet_preprocessed'] = nnUNet_preprocessed
    os.environ['nnUNet_results'] = nnUNet_results

def run_nnunet_predict(input_folder, output_folder_2d, output_folder_3d):
    """Run nnUNet predictions for 2D and 3D models."""
    configure_environment()
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        # Run prediction for 2D model
        logger.info("Running prediction for 2D model...")
        subprocess.run([
            'nnUNetv2_predict',
            '-d', 'Dataset006_totalbleed',
            '-i', input_folder,
            '-o', output_folder_2d,
            '-f', '0', '1', '2', '3', '4',
            '-tr', 'nnUNetTrainer',
            '-c', '2d',
            '-p', 'nnUNetPlans',
            '--save_probabilities'
        ], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Run prediction for 3D_lowres model
        logger.info("Running prediction for 3D_lowres model...")
        subprocess.run([
            'nnUNetv2_predict',
            '-d', 'Dataset006_totalbleed',
            '-i', input_folder,
            '-o', output_folder_3d,
            '-f', '0', '1', '2', '3', '4',
            '-tr', 'nnUNetTrainer',
            '-c', '3d_lowres',
            '-p', 'nnUNetPlans',
            '--save_probabilities'
        ], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during nnUNet prediction: {e}")

def run_ensemble(output_folder_2d, output_folder_3d, ensemble_output_folder):
    """Run ensemble of 2D and 3D model predictions."""
    env = os.environ.copy()
    try:
        logger.info("Running ensemble of predictions...")
        subprocess.run([
            'nnUNetv2_ensemble',
            '-i', output_folder_2d, output_folder_3d,
            '-o', ensemble_output_folder,
            '-np', '8'
        ], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        logger.info("Ensemble completed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during ensembling: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run nnUNet Bleeding Segmentation')
    parser.add_argument('-i', '--input_folder', required=True, help='Input folder path')
    parser.add_argument('-o', '--output_folder', required=True, help='Final output folder path')

    args = parser.parse_args()

    output_folder_2d = os.path.join(args.output_folder, "output_model_2d")
    output_folder_3d = os.path.join(args.output_folder, "output_model_3d_lowres")
    ensemble_output_folder = os.path.join(args.output_folder, "ensemble_output")

    # Run the prediction for 2D and 3D models
    run_nnunet_predict(args.input_folder, output_folder_2d, output_folder_3d)

    # Run the ensemble of both models
    run_ensemble(output_folder_2d, output_folder_3d, ensemble_output_folder)
