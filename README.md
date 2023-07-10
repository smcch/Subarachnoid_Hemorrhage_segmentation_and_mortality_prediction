SAH Mortality Prediction Repository

This repository provides a pipeline for predicting mortality in patients with subarachnoid hemorrhage (SAH) using the AUCMEDI model. The pipeline includes preprocessing of DICOM files, running the AUCMEDI prediction model, and generating a report with prediction results.
Installation

    Clone the repository to your local machine:

    shell

git clone https://github.com/your-username/sah-mortality-prediction.git

Install the required dependencies. Make sure you have Python 3.7 or later installed. Use the following command to install the dependencies:

shell

    pip install -r requirements.txt

    Note: Additional dependencies such as FSL, ANTs, and dcm2niix may need to be installed separately. Please refer to their respective documentation for installation instructions.

    Download the AUCMEDI model file (3D.DenseNet121.model.best.loss.hdf5) and place it in the root directory of the repository.

Usage

    Prepare your input data:
        Create a directory containing the DICOM files of the SAH patients.
        Each patient should have a separate subdirectory with their DICOM files.
        Ensure that the DICOM files contain the necessary information for the prediction (e.g., CT scans).

    Run the SAH Mortality Prediction pipeline:

    shell

    python SAH_mortality_prediction.py -i /path/to/input -o /path/to/output --model 3D.DenseNet121.model.best.loss.hdf5

    Replace /path/to/input with the path to the directory containing the DICOM files, and /path/to/output with the desired output directory. The --model argument specifies the path to the AUCMEDI model file.

    View the results:
        The pipeline will generate processed NIfTI files and a predictions.csv file in the output directory.
        Additionally, a report in PDF format will be created for each patient, providing the prediction results, volume slices, and XAI slices.
        The report files will be named as report_<subject_id>.pdf.

License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
Contact

For any questions or inquiries, please contact:

    Santiago Cepeda: santiago.cepeda@example.com
    Dominik Müller: dominik.mueller@example.com

Feel free to reach out if you have any issues or suggestions!

That's an improved format for the README file using Markdown. You can further customize and expand it to meet your specific needs. Ensure to replace the placeholders (/path/to/input, /path/to/output, your-username, etc.) with the actual paths and information relevant to your project.
