import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from SAH_mortality_prediction import preprocess, predict_aucmedi, generate_report

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
    model_path = '3D.DenseNet121.model.best.loss.hdf5' # Replace with the actual path to the model file

    # Preprocess the input data
    preprocess(input_path, output_path)

    # Run the AUCMEDI prediction
    for patient_folder in os.listdir(output_path):
        patient_dir = os.path.join(output_path, patient_folder)
        predict_aucmedi(patient_dir, model_path, output_path, output_path)

        # Generate the report
        subject_id = patient_folder.split("_")[-1]
        volume_nifti = os.path.join(patient_dir, f"{subject_id}_ct.nii.gz")
        xai_nifti = os.path.join(output_path, f"{subject_id}_ct.nii.gz")
        probability = get_probability(patient_dir)
        generate_report(output_path, subject_id, volume_nifti, xai_nifti, probability)

def get_probability(patient_dir):
    # Read the predictions.csv file
    predictions_file = os.path.join(patient_dir, 'predictions.csv')
    df_merged = pd.read_csv(predictions_file)

    # Get the probability of death
    probability = df_merged.iloc[0]['pd_ASH:1'] * 100

    return probability

# Create the main window
window = tk.Tk()
window.title("SAH Mortality Prediction")
window.geometry("400x200")

# Create input directory selection
input_label = tk.Label(window, text="Input Directory:")
input_label.pack()
input_entry = tk.Entry(window, width=50)
input_entry.pack()
input_button = tk.Button(window, text="Select", command=select_input_directory)
input_button.pack()

# Create output directory selection
output_label = tk.Label(window, text="Output Directory:")
output_label.pack()
output_entry = tk.Entry(window, width=50)
output_entry.pack()
output_button = tk.Button(window, text="Select", command=select_output_directory)
output_button.pack()

# Create prediction button
predict_button = tk.Button(window, text="Run Prediction", command=run_prediction)
predict_button.pack()

# Start the GUI event loop
window.mainloop()
