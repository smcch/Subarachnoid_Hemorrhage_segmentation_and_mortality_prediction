# ==============================================================================#
#  Author:       Santiago Cepeda, Dominik Müller                                #
#  Copyright:    Río Hortega University Hospital in Valladolid, Spain           #
#                University of Augsburg, Germany                                #
# ==============================================================================#

import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from SAH_mortality_prediction import preprocess, predict_aucmedi, generate_report
from inference_2 import main as inference_main


def select_input_directory():
    input_dir = filedialog.askdirectory()
    input_entry.delete(0, tk.END)
    input_entry.insert(0, input_dir)


def select_output_directory():
    output_dir = filedialog.askdirectory()
    output_entry.delete(0, tk.END)
    output_entry.insert(0, output_dir)


def run_prediction():
    input_path = input_entry.get()
    output_path = output_entry.get()
    model_path = '3D.DenseNet121.model.best.loss.hdf5'  # Replace with the actual path to the model file

    # Preprocess the input data
    preprocess(input_path, output_path)

    # Run the AUCMEDI prediction and segmentation
    for patient_folder in os.listdir(output_path):
        patient_dir = os.path.join(output_path, patient_folder)

        # Run the AUCMEDI prediction
        predict_aucmedi(patient_dir, model_path, output_path, output_path)

        # Extract the subject ID from the patient folder name
        subject_id = patient_folder.split("_")[-1]

        # Run the inference
        inference_main(patient_dir, patient_dir)

        # Generate the report
        volume_nifti = os.path.join(patient_dir, f"{subject_id}_ct.nii.gz")
        xai_nifti = os.path.join(patient_dir, 'aucmedi', f"{subject_id}_ct.nii.gz")
        probability = get_probability(patient_dir)

        # Define the correct path to the segmentation_nifti file
        segmentation_nifti = os.path.join(patient_dir, "segmentations", f"output_{subject_id}_ct", f"second_channel_{subject_id}_ct.nii.gz")

        generate_report(patient_dir, subject_id, volume_nifti, xai_nifti, probability, segmentation_nifti)

    # Update status and show "Done" message
    status_label.config(text="Status: Done")
    message_label.config(text="Prediction and segmentation completed successfully.")




def get_probability(patient_dir):
    # Read the predictions.csv file
    predictions_file = os.path.join(patient_dir, 'aucmedi', 'predictions.csv')  # CHANGE: Read from 'aucmedi' subfolder
    df_merged = pd.read_csv(predictions_file)

    # Get the probability of death
    probability = df_merged.iloc[0]['pd_ASH:1'] * 100

    return probability


def exit_app():
    window.destroy()


# Create the main window
window = tk.Tk()
window.title("SAH Segmentation and Mortality Prediction")

# Set a ttk theme for a nicer appearance
style = ttk.Style()
style.theme_use("clam")

# Load and display the logo image
logo_path = os.path.abspath("unvrh.png")
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
    # Resize the image
    desired_size = (326, 241)  # (width, height)
    logo_image = logo_image.resize(desired_size, Image.BICUBIC)
    logo_photo = ImageTk.PhotoImage(logo_image)
    logo_label = ttk.Label(window, image=logo_photo)
    logo_label.pack()

# Create input directory selection
input_label = ttk.Label(window, text="Input Directory:")
input_label.pack()
input_entry = ttk.Entry(window, width=50)
input_entry.pack()
input_button = ttk.Button(window, text="Select", command=select_input_directory)
input_button.pack()

# Create output directory selection
output_label = ttk.Label(window, text="Output Directory:")
output_label.pack()
output_entry = ttk.Entry(window, width=50)
output_entry.pack()
output_button = ttk.Button(window, text="Select", command=select_output_directory)
output_button.pack()

# Create prediction button
predict_button = ttk.Button(window, text="Run Prediction and Segmentation", command=run_prediction)
predict_button.pack()

# Create exit button
exit_button = ttk.Button(window, text="Exit", command=exit_app)
exit_button.pack()

# Create status bar
status_label = ttk.Label(window, text="Status: Idle")
status_label.pack(side=tk.LEFT)
message_label = ttk.Label(window, text="")
message_label.pack(side=tk.RIGHT)

# Fit window size to content
window.update_idletasks()
window.geometry(f"{window.winfo_reqwidth()}x{window.winfo_reqheight()}")

# Start the GUI event loop
window.mainloop()
