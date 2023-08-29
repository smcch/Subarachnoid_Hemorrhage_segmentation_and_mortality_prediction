# Subarachnoid Hemorrhage Automatic Segmentation and Mortality Prediction

This repository contains the Python implementation of the papers: 

- "Enhanced Mortality Prediction In Patients With Subarachnoid Haemorrhage Using A Deep Learning Model Based On The Initial CT Scan" Sergio García-García, Santiago Cepeda Chafla, Dominik Müller, Alejandra Mosteiro Cadaval, Ramón Torné-Torné, Silvia Agudo, Natalia de la Torre, Ignacio Arrese, Rosario Sarabia. Pre-print. arXiv:2308.13373  https://doi.org/10.48550/arXiv.2308.13373
- "An End-To-End Pipeline for Fully Automatic Subarachnoid Hemorrhage Segmentation in CT scans using Swin Transformers". Sergio García-García, Santiago Cepeda Chafla, Ignacio Arrese and Rosario Sarabia. 

The repository introduces a streamlined pipeline for:
    Automatic Bleeding Segmentation: This segment utilizes the MONAI framework to perform segmentation on non-contrast CT scans from patients diagnosed with subarachnoid hemorrhage.
    Mortality Risk Prediction: Predicts the mortality risk within a span of 3 months post-admission using the AUCMEDI framework.

The comprehensive pipeline encompasses DICOM file preprocessing, automatic segmentation using a Vision Transformer (ViT)-based model, mortality prediction through the AUCMEDI model, and the generation of a detailed report. This report includes prediction outcomes, segmentation files, and a volumetric analysis of the bleeding.

![imagen](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction/assets/87584415/bd1e2bda-c48f-42de-8e96-a417c920389f)


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


### Command Line Execution for mortality prediction task

Prepare your input data:

Create a directory containing the DICOM files of the SAH patients.
Each patient should have a separate subdirectory with their DICOM files.
Ensure that the DICOM files contain the necessary information for the prediction (e.g., CT scans).

Run the SAH Mortality Prediction pipeline:

```
python SAH_mortality_prediction.py -i /path/to/input -o /path/to/output --model 3D.DenseNet121.model.best.loss.hdf5 -g 0
```

Replace /path/to/input with the path to the directory containing the DICOM files, and /path/to/output with the desired output directory. The --model argument specifies the path to the AUCMEDI model file.

### Command Line Execution for automatic segmetation

Setting Up the Pretrained Model:

Begin by downloading the pretrained model for the pipeline.

Click on the link below to access the model:

[Download Pretrained Model](https://drive.google.com/file/d/1ChgWWranUdj6w3NXXy_RDP2gRQTMNc35/view?usp=drive_link)

Ensure you place the downloaded model in the appropriate directory as mentioned in subsequent steps or as required by the pipeline.

Expected Input Format:
Ensure your processed non-contrast CT scan in NIfTI format is structured as follows:

```
input_folder/
└── subject_ID/
    └── subject_ID_ct.nii.gz
```

Then run the comman:

```
python inference_2.py --input_dir --output_dir
```

### Graphical User Interface (GUI) Execution

Alternatively, you can use the graphical user interface (GUI) for execution of both tasks (segmentation and mortality prediction). Run the following command:
```
python gui.py
```
This will launch the GUI, allowing you to interactively select the input and output directories without specify the AUCMEDI model file.


![imagen](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction/assets/87584415/5f97192e-3ce1-4a87-9224-ee319acc2ad1)



View the results:

The pipeline will generate processed NIfTI files and a predictions.csv file in the output directory.
Additionally, a report in PDF format will be created for each patient, providing the prediction results, volume slices, and XAI slices.
The report files will be named as report_<subject_id>.pdf.

![imagen](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction/assets/87584415/b4be72a5-b53a-4c12-b7c5-6a7c67882a81)

![imagen](https://github.com/smcch/Subarachnoid_Hemorrhage_segmentation_and_mortality_prediction/assets/87584415/3c6a2c09-cac3-416f-9b7c-5a5a1695bbdb)



## Citations

If you find this pipeline useful for your academic purposes, please include the following citations:

- DICOM to NiFTI converter: `dcm2niix`, available at https://github.com/rordenlab/dcm2niix/releases/tag/v1.0.20220720
	- Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data analysis: DICOM to NIfTI conversion. J Neurosci Methods. 2016;264:47-56. doi:10.1016/j.jneumeth.2016.03.001.
- AUCMEDI: a framework for Automated Classification of Medical Images (Version X.Y.Z) [Computer software].
  - Müller, D., Mayer, S., Hartmann, D., Schneider, P., Soto-Rey, I., & Kramer, F. (2022). https://doi.org/10.5281/zenodo.6633540. GitHub repository. https://github.com/frankkramer-lab/aucmedi
- ANTsPy. Advanced Normalization Tools in Python. https://github.com/ANTsX/ANTsPy
- FMRIB Software Library v6.0 - BET (Brain Extraction Tool). https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET
  - M. Jenkinson, M. Pechaud, and S. Smith. BET2: MR-based estimation of brain, skull and scalp surfaces. In Eleventh Annual Meeting of the Organization for Human Brain Mapping, 2005.
- MONAI: Medical Open Network for Artificial Intelligence https://zenodo.org/record/8018287	https://monai.io/

## License
Creative Commons Attribution-NonCommercial License: This repository is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) license. This license allows others to freely use, modify, and distribute the software for non-commercial purposes only. You are granted the right to use this software for personal, educational, and non-profit projects, but commercial use is not permitted without explicit permission. For more details, please refer to the LICENSE file.

## Contact

For any questions or inquiries, please contact:

Santiago Cepeda: scepedac@saludcastillayleon.es
