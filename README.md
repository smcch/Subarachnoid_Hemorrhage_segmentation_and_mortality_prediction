# SAH Mortality Prediction

This repository provides a pipeline for predicting mortality in patients with subarachnoid hemorrhage (SAH) using the AUCMEDI framework. The pipeline includes preprocessing of DICOM files, running the AUCMEDI prediction model, and generating a report with prediction results.

## Installation

1. Clone the repository to your local machine:
git clone https://github.com/your-username/sah-mortality-prediction.git

2. Install the required dependencies. Make sure you have Python 3.7 or later installed. Use the following command to install the dependencies:
pip install -r requirements.txt
   **Note**: Additional dependencies such as FSL, ANTs, and dcm2niix may need to be installed separately. Please refer to their respective documentation for installation instructions.

3. Download the AUCMEDI model file (`3D.DenseNet121.model.best.loss.hdf5`) and place it in the root directory of the repository.

## Usage

1. **Prepare your input data:**

   - Create a directory containing the DICOM files of the SAH patients.
   - Each patient should have a separate subdirectory with their DICOM files.
   - Ensure that the DICOM files contain the necessary information for the prediction (e.g., CT scans).

2. **Run the SAH Mortality Prediction pipeline:**
python SAH_mortality_prediction.py -i /path/to/input -o /path/to/output --model 3D.DenseNet121.model.best.loss.hdf5
   Replace `/path/to/input` with the path to the directory containing the DICOM files, and `/path/to/output` with the desired output directory. The `--model` argument specifies the path to the AUCMEDI model file.

3. **View the results:**

   - The pipeline will generate processed NIfTI files and a `predictions.csv` file in the output directory.
   - Additionally, a report in PDF format will be created for each patient, providing the prediction results, volume slices, and XAI slices.
   - The report files will be named as `report_<subject_id>.pdf`.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or inquiries, please contact:

- Santiago Cepeda: [santiago.cepeda@example.com](mailto:scepedac@saludcastillayleon.es)
