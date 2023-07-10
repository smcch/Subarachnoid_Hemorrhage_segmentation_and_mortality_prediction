SAH Mortality Prediction

This repository provides a pipeline for predicting mortality in patients with subarachnoid hemorrhage (SAH) using the AUCMEDI framework. The pipeline includes preprocessing of DICOM files, running the AUCMEDI prediction model, and generating a report with prediction results.

Installation

Download the Miniconda installer:
arduino
Copy code
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
Run the installer script:
Copy code
bash Miniconda3-latest-MacOSX-x86_64.sh
Initialize Miniconda:
bash
Copy code
source ~/.bash_profile
Create a virtual environment:
lua
Copy code
conda create --name myenv
Replace 'myenv' with your desired name.

Activate the virtual environment:
Copy code
conda activate myenv
Clone the repository to your local machine:
bash
Copy code
git clone https://github.com/your-username/sah-mortality-prediction.git
Navigate to the directory where the repository was cloned.
Install the required dependencies. Make sure you have Python 3.7 or later installed. Use the following command to install the dependencies:
Copy code
pip install -r requirements.txt
Note: Additional dependencies such as FSL, ANTs, and dcm2niix may need to be installed separately. Please refer to their respective documentation for installation instructions.

Usage

Prepare your input data:
Create a directory containing the DICOM files of the SAH patients.
Each patient should have a separate subdirectory with their DICOM files.
Ensure that the DICOM files contain the necessary information for the prediction (e.g., CT scans).
Run the SAH Mortality Prediction pipeline:
css
Copy code
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
