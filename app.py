import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from preprocessing import preprocess_data
from mlp import MultiLayerPerceptron
from matrices import compute_confusion_matrix, compute_accuracy


# -- Globals to hold the trained model and scaler ---
trained_mlp     = None
trained_scaler  = None
trained_species = None

def start_training_process(entries, display_widgets):
    global trained_mlp, trained_scaler, trained_species
    try:
        # Get values from GUI
        lr       = float(entries['lr'].get())
        epochs   = int(entries['epochs'].get())
        act_type = entries['act'].get()
        use_bias = entries['bias'].get()
        h_layers = [int(n.strip()) for n in entries['layers'].get().split(",")]

        # Load and preprocess data
        X_train, y_train, X_test, y_test, species_list, scaler = preprocess_data()

        # Initialize MLP
        mlp = MultiLayerPerceptron(
            input_size=5,
            hidden_layers=h_layers,
            output_size=3,
            learning_rate=lr,
            activation_type=act_type,
            use_bias=use_bias
        )

        # Train the model
        mlp.train(X_train, y_train, epochs=epochs)
        
        # Calculate Training Accuracy
        y_pred_train = mlp.test(X_train)
        train_acc = compute_accuracy(y_train, y_pred_train)
        
        # Calculate Testing Accuracy
        y_pred_test = mlp.test(X_test)
        test_acc = compute_accuracy(y_test, y_pred_test)

        # Calculate Confusion Matrix
        matrix = compute_confusion_matrix(y_test, y_pred_test)

        # Save trained objects
        trained_mlp     = mlp
        trained_scaler  = scaler
        trained_species = species_list

        # Update GUI Results
        display_widgets['train_acc_label'].config(text=f"Train Accuracy: {train_acc:.2f} %", fg="#34495e")
        display_widgets['acc_label'].config(text=f"Test Accuracy: {test_acc:.2f} %", fg="#27ae60")

        # Display confusion matrix 
        display_widgets['matrix_box'].delete("1.0", tk.END)
        header = f"{'Actual/Pred':<12} | " + " | ".join([f"{s[:6]:>7}" for s in species_list]) + "\n"
        display_widgets['matrix_box'].insert(tk.END, header + "-"*45 + "\n")
        for i, row in enumerate(matrix):
            row_str = f"{species_list[i][:10]:<12} | " + " | ".join([f"{val:>7}" for val in row]) + "\n"
            display_widgets['matrix_box'].insert(tk.END, row_str)

    except Exception as e:
        messagebox.showerror("Runtime Error", f"Something went wrong:\n{e}")

def classify_sample(sample_entries, result_label):
    global trained_mlp, trained_scaler, trained_species
    if trained_mlp is None:
        messagebox.showwarning("Not Trained", "Please train the model first!")
        return
    try:
        location_mapping = {'Torgersen': 0.0, 'Biscoe': 1.0, 'Dream': 2.0}
        culmen_length  = float(sample_entries['culmen_length'].get())
        culmen_depth   = float(sample_entries['culmen_depth'].get())
        flipper_length = float(sample_entries['flipper_length'].get())
        location_str   = sample_entries['location'].get()
        body_mass      = float(sample_entries['body_mass'].get())

        location_val = location_mapping[location_str]
        sample = np.array([[culmen_length, culmen_depth, flipper_length, location_val, body_mass]])
        sample_scaled = trained_scaler.transform(sample)

        _, output = trained_mlp.forward_propagation(sample_scaled)
        pred_index = int(np.argmax(output, axis=1)[0])
        result_label.config(text=f"Predicted Species: {trained_species[pred_index]}", fg="#2980b9")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def run_gui():
    root = tk.Tk()
    root.title("Penguin Classifier - Task 2")
    
    # Standardized window size for 850px width to fit two columns
    root.geometry("850x650")
    root.configure(bg="#f8f9fa")

    # Title at the top
    tk.Label(root, text="MLP PENGUIN CLASSIFIER", font=("Arial", 18, "bold"),
             bg="#f8f9fa", pady=20).pack()

    # Main Container for the two columns
    main_frame = tk.Frame(root, bg="#f8f9fa")
    main_frame.pack(fill="both", expand=True, padx=20)

    # --- LEFT COLUMN (Training & Settings) ---
    left_column = tk.Frame(main_frame, bg="#f8f9fa")
    left_column.grid(row=0, column=0, sticky="nw", padx=10)

    # 1. Network Settings Frame
    input_frame = tk.LabelFrame(left_column, text=" 1. Network Settings ", padx=15, pady=15)
    input_frame.pack(pady=5, fill="x")

    tk.Label(input_frame, text="Neurons (e.g. 8,4):").grid(row=0, column=0, sticky="w", pady=3)
    layer_entry = ttk.Entry(input_frame, width=15); layer_entry.insert(0, "8"); layer_entry.grid(row=0, column=1, padx=10)

    tk.Label(input_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w", pady=3)
    lr_entry = ttk.Entry(input_frame, width=15); lr_entry.insert(0, "0.1"); lr_entry.grid(row=1, column=1, padx=10)

    tk.Label(input_frame, text="Epochs:").grid(row=2, column=0, sticky="w", pady=3)
    epoch_entry = ttk.Entry(input_frame, width=15); epoch_entry.insert(0, "300"); epoch_entry.grid(row=2, column=1, padx=10)

    tk.Label(input_frame, text="Activation:").grid(row=3, column=0, sticky="w", pady=3)
    act_var = tk.StringVar(value="Sigmoid")
    ttk.Combobox(input_frame, textvariable=act_var, values=["Sigmoid", "Tanh"],
                  state="readonly", width=12).grid(row=3, column=1, padx=10)

    bias_var = tk.BooleanVar(value=True)
    tk.Checkbutton(input_frame, text="Use Bias Term", variable=bias_var).grid(row=4, columnspan=2, pady=5)

    # Train button
    entries = {'layers': layer_entry, 'lr': lr_entry, 'epochs': epoch_entry, 'act': act_var, 'bias': bias_var}
    
    # 2. Results Frame (Below Settings)
    res_frame = tk.LabelFrame(left_column, text=" 2. Results ", padx=15, pady=10)
    res_frame.pack(pady=10, fill="x")

    train_acc_l = tk.Label(res_frame, text="Train Accuracy: -- %", font=("Arial", 10), fg="#34495e")
    train_acc_l.pack()
    test_acc_l = tk.Label(res_frame, text="Test Accuracy: -- %", font=("Arial", 11, "bold"), fg="#27ae60")
    test_acc_l.pack()

    m_box = tk.Text(res_frame, height=6, width=45, font=("Courier", 9), padx=5, pady=5)
    m_box.pack(pady=5)

    displays = {'train_acc_label': train_acc_l, 'acc_label': test_acc_l, 'matrix_box': m_box}

    tk.Button(left_column, text="START TRAINING & EVALUATE", bg="#2980b9", fg="white",
              font=("Arial", 11, "bold"), pady=10,
              command=lambda: start_training_process(entries, displays)).pack(pady=10, fill="x")


    # --- RIGHT COLUMN (Single Sample Classification) ---
    right_column = tk.Frame(main_frame, bg="#f8f9fa")
    right_column.grid(row=0, column=1, sticky="nw", padx=10)

    cls_frame = tk.LabelFrame(right_column, text=" 3. Classify Sample ", padx=20, pady=20)
    cls_frame.pack(pady=5, fill="both", expand=True)

    fields = [
        ("Culmen Length:", "culmen_length", "50.0"), 
        ("Culmen Depth:", "culmen_depth", "19.5"), 
        ("Flipper Length:", "flipper_length", "196.0"), 
        ("Body Mass:", "body_mass", "3900.0")
    ]
    
    sample_entries = {}
    for r, (label, key, default_val) in enumerate(fields):
        tk.Label(cls_frame, text=label).grid(row=r, column=0, sticky="e", pady=8)
        e = ttk.Entry(cls_frame, width=20)
        e.insert(0, default_val) 
        e.grid(row=r, column=1, padx=15, pady=8, sticky="w")
        sample_entries[key] = e

    tk.Label(cls_frame, text="Location:").grid(row=4, column=0, sticky="e", pady=8)
    loc_var = tk.StringVar(value="Dream") 
    ttk.Combobox(cls_frame, textvariable=loc_var, values=["Torgersen", "Biscoe", "Dream"],
                 state="readonly", width=17).grid(row=4, column=1, padx=15, pady=8, sticky="w")
    sample_entries['location'] = loc_var

    result_lbl = tk.Label(cls_frame, text="Predicted: --", font=("Arial", 12, "bold"), fg="#7f8c8d")
    result_lbl.grid(row=5, columnspan=2, pady=20)

    tk.Button(cls_frame, text="CLASSIFY", bg="#27ae60", fg="white", font=("Arial", 11, "bold"), 
              width=15, pady=8,
              command=lambda: classify_sample(sample_entries, result_lbl)).grid(row=6, columnspan=2, pady=10)

    root.mainloop()

if __name__ == "__main__":
    run_gui()