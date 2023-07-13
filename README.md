# SAH Mortality Prediction

This repository provides a pipeline for predicting mortality in patients with subarachnoid hemorrhage (SAH) using the AUCMEDI framework. The pipeline includes preprocessing of DICOM files, running the AUCMEDI prediction model, and generating a report with prediction results.

## Installation

Download the Miniconda installer:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
```

Run the installer script:

```
bash Miniconda3-latest-MacOSX-x86_64.sh
```

Initialize Miniconda:

```
source ~/.bash_profile
```

Create a virtual environment:

```
conda create --name myenv
```

Activate the virtual environment:

```
conda activate myenv
```

Replace 'myenv' with your desired name.

Clone the repository to your local machine:

```
git clone https://github.com/your-username/sah-mortality-prediction.git
```

Navigate to the directory where the repository was cloned.

Install the required dependencies. Make sure you have Python 3.7 or later installed. Use the following command to install the dependencies:

```
pip install -r requirements.txt
```


Note: Additional dependencies such as FSL, ANTs, and dcm2niix need to be installed separately and added to the system path. Please follow the instructions below to install them:

### Installing External Dependencies

#### dcm2niix

1. Visit the dcm2niix repository: [https://github.com/rordenlab/dcm2niix](https://github.com/rordenlab/dcm2niix)
2. Follow the installation instructions provided in the repository to install dcm2niix.
3. Add dcm2niix to the system path.

#### FSL

1. Visit the FSL installation page: [https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation)
2. Follow the installation instructions provided to install FSL.
3. Add FSL to the system path.

## Usage

For Mac OS users: to avoid error reading .DS_Store files, please enter this command line to delet those files from the input and output directories:

```
find /path/to/directory -name .DS_Store -type f -delete
```

replace /path/to/directory  with the actual path of the input and output directory.


### Command Line Execution

Prepare your input data:

Create a directory containing the DICOM files of the SAH patients.
Each patient should have a separate subdirectory with their DICOM files.
Ensure that the DICOM files contain the necessary information for the prediction (e.g., CT scans).

Run the SAH Mortality Prediction pipeline:

```
python SAH_mortality_prediction.py -i /path/to/input -o /path/to/output --model 3D.DenseNet121.model.best.loss.hdf5 -g 0
```

Replace /path/to/input with the path to the directory containing the DICOM files, and /path/to/output with the desired output directory. The --model argument specifies the path to the AUCMEDI model file.

### Graphical User Interface (GUI) Execution

Alternatively, you can use the graphical user interface (GUI) for execution. Run the following command:
```
python gui.py
```
This will launch the GUI, allowing you to interactively select the input and output directories without specify the AUCMEDI model file.


<img width="452" alt="Captura de pantalla 2023-07-12 a las 10 30 09" src="https://github.com/smcch/Subarachnoid_Hemorrhage_mortality_prediction/assets/87584415/27f2057e-c0f0-411e-a723-7c62b1427d44">


View the results:

The pipeline will generate processed NIfTI files and a predictions.csv file in the output directory.
Additionally, a report in PDF format will be created for each patient, providing the prediction results, volume slices, and XAI slices.
The report files will be named as report_<subject_id>.pdf.

<img width="331" alt="Captura de pantalla 2023-07-12 a las 10 31 11" src="https://github.com/smcch/Subarachnoid_Hemorrhage_mortality_prediction/assets/87584415/ff45fd4b-1b24-442b-a69a-013642255de7">

<img width="321" alt="Captura de pantalla 2023-07-12 a las 10 31 23" src="https://github.com/smcch/Subarachnoid_Hemorrhage_mortality_prediction/assets/87584415/62a56fc0-cdc3-48fa-b922-96a4990db8ef">


## License

This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.

## Contact

For any questions or inquiries, please contact:

Santiago Cepeda: scepedac@saludcastillayleon.es
