import tkinter as tk
from tkinter import ttk
import pandas as pd
import joblib
import numpy as np

# Load the trained model using joblib (Make sure to train and save the model first)
model = joblib.load("Disease-Prediction-from-Symptoms-model.pkl")

# Load the preprocessed data
df = pd.read_csv("training_dataset.csv")

# Create a list to store selected symptom indices
selected_symptoms = []

# Create a function to predict the disease based on symptoms
def predict_disease():
    if not selected_symptoms:
        result_label.config(text="Please select symptoms.")
    else:
        result_label.config(text="")  # Clear the result label
        input_data = np.zeros(len(df.columns) - 1)  # Initialize an array of zeros
        for i in selected_symptoms:
            input_data[i] = 1  # Set the corresponding symptom to 1 if selected
        disease = model.predict([input_data])
        result_label.config(text=f"Predicted Disease: {disease[0]}")

# Create a function to update the selected symptoms list
def update_selected_symptoms(event):
    global selected_symptoms
    selected_symptoms = [symptom_list.index(symptom_listbox.get(i)) for i in symptom_listbox.curselection()]

# Create a function to update the symptom list based on the search
def update_symptom_list(event):
    search_text = symptom_search_entry.get().strip().lower()
    matching_indices = [
        i for i, symptom in enumerate(symptom_list) if search_text in symptom.lower()
    ]
    symptom_listbox.delete(0, tk.END)  # Clear the listbox
    for i in matching_indices:
        symptom_listbox.insert(tk.END, symptom_list[i])

# Create a function to clear the selected symptoms and result
def clear_selection():
    symptom_listbox.selection_clear(0, tk.END)  # Clear selected symptoms
    result_label.config(text="")  # Clear the result label

# Create the main GUI window
root = tk.Tk()
root.title("Disease Prediction GUI")

# Create a frame to hold the symptom selection
symptom_frame = ttk.LabelFrame(root, text="Select Symptoms")
symptom_frame.grid(column=0, row=0, padx=10, pady=10, sticky="nsew")

# Create an Entry widget for searching symptoms
symptom_search_entry = ttk.Entry(symptom_frame)
symptom_search_entry.grid(column=0, row=0, padx=10, pady=5, sticky="w")
symptom_search_entry.bind("<KeyRelease>", update_symptom_list)

# Create a Listbox for symptom selection (multiple selections)
symptom_list = df.columns[1:].tolist()
symptom_listbox = tk.Listbox(symptom_frame, selectmode=tk.MULTIPLE)
for symptom in symptom_list:
    symptom_listbox.insert(tk.END, symptom)
symptom_listbox.grid(column=0, row=1, padx=10, pady=5, sticky="nsew")
symptom_listbox.bind("<<ListboxSelect>>", update_selected_symptoms)

# Create a button to predict the disease
predict_button = ttk.Button(root, text="Predict Disease", command=predict_disease)
predict_button.grid(column=0, row=1, padx=10, pady=10)

# Create a "Clear" button to reset the selection and result
clear_button = ttk.Button(root, text="Clear Selection", command=clear_selection)
clear_button.grid(column=0, row=2, padx=10, pady=10)

# Create a label to display the prediction result
result_label = ttk.Label(root, text="")
result_label.grid(column=0, row=3, padx=10, pady=10)

# Configure a flexible layout to adapt to window size changes
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
symptom_frame.columnconfigure(0, weight=1)
symptom_frame.rowconfigure(1, weight=1)

root.mainloop()
