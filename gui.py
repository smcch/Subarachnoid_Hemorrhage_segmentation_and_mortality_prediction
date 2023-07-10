#==============================================================================#
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
#==============================================================================#
import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from SAH_mortality_prediction import preprocess, predict_aucmedi, generate_report

import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
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

    # Update status and show "Done" message
    status_label.config(text="Status: Done")
    message_label.config(text="Prediction completed successfully.")

def get_probability(patient_dir):
    # Read the predictions.csv file
    predictions_file = os.path.join(patient_dir, 'predictions.csv')
    df_merged = pd.read_csv(predictions_file)

    # Get the probability of death
    probability = df_merged.iloc[0]['pd_ASH:1'] * 100

    return probability

def exit_app():
    window.destroy()

# Create the main window
window = tk.Tk()
window.title("SAH Mortality Prediction")

# Set a ttk theme for a nicer appearance
style = ttk.Style()
style.theme_use("clam")

# Load and display the logo image
logo_path = os.path.abspath("unvrh.png")
if os.path.exists(logo_path):
    logo_image = Image.open(logo_path)
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
predict_button = ttk.Button(window, text="Run Prediction", command=run_prediction)
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
