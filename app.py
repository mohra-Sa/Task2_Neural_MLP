import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from preprocessing import preprocess_data
from mlp import MultiLayerPerceptron
from matrices import compute_confusion_matrix, compute_accuracy

# ── Globals to hold the trained model and scaler ──────────────────────────────
trained_mlp     = None
trained_scaler  = None
trained_species = None
# ─────────────────────────────────────────────────────────────────────────────


def start_training_process(entries, display_widgets):
    global trained_mlp, trained_scaler, trained_species
    try:
        lr       = float(entries['lr'].get())
        epochs   = int(entries['epochs'].get())
        act_type = entries['act'].get()
        use_bias = entries['bias'].get()
        h_layers = [int(n.strip()) for n in entries['layers'].get().split(",")]

        X_train, y_train, X_test, y_test, species_list, scaler = preprocess_data()

        mlp = MultiLayerPerceptron(
            input_size=5,
            hidden_layers=h_layers,
            output_size=3,
            learning_rate=lr,
            activation_type=act_type,
            use_bias=use_bias
        )

        mlp.train(X_train, y_train, epochs=epochs)
        y_pred = mlp.test(X_test)

        acc    = compute_accuracy(y_test, y_pred)
        matrix = compute_confusion_matrix(y_test, y_pred)

        # Save trained objects for single-sample classification
        trained_mlp     = mlp
        trained_scaler  = scaler
        trained_species = species_list

        display_widgets['acc_label'].config(text=f"Accuracy: {acc:.2f} %", fg="#27ae60")

        display_widgets['matrix_box'].delete("1.0", tk.END)
        header = f"{'Actual/Pred':<12} | " + " | ".join([f"{s[:6]:>8}" for s in species_list]) + "\n"
        display_widgets['matrix_box'].insert(tk.END, header + "-"*45 + "\n")
        for i, row in enumerate(matrix):
            row_str = f"{species_list[i][:10]:<12} | " + " | ".join([f"{val:>8}" for val in row]) + "\n"
            display_widgets['matrix_box'].insert(tk.END, row_str)

    except Exception as e:
        messagebox.showerror("Runtime Error", f"Something went wrong:\n{e}")


def classify_sample(sample_entries, result_label):
    """Classify a single manually-entered sample using the trained MLP."""
    global trained_mlp, trained_scaler, trained_species

    if trained_mlp is None:
        messagebox.showwarning("Not Trained", "Please train the model first before classifying a sample.")
        return

    try:
        location_mapping = {'Torgersen': 0.0, 'Biscoe': 1.0, 'Dream': 2.0}

        culmen_length  = float(sample_entries['culmen_length'].get())
        culmen_depth   = float(sample_entries['culmen_depth'].get())
        flipper_length = float(sample_entries['flipper_length'].get())
        location_str   = sample_entries['location'].get()
        body_mass      = float(sample_entries['body_mass'].get())

        if location_str not in location_mapping:
            messagebox.showerror("Input Error", "Location must be: Torgersen, Biscoe, or Dream")
            return

        location_val = location_mapping[location_str]

        # Build raw feature vector in the same order as training:
        # ['CulmenLength', 'CulmenDepth', 'FlipperLength', 'OriginLocation', 'BodyMass']
        sample = np.array([[culmen_length, culmen_depth, flipper_length, location_val, body_mass]])

        # Apply the same scaler fitted on training data
        sample_scaled = trained_scaler.transform(sample)

        # Forward pass → argmax → species name
        _, output = trained_mlp.forward_propagation(sample_scaled)
        pred_index = int(np.argmax(output, axis=1)[0])
        predicted_species = trained_species[pred_index]

        result_label.config(
            text=f"Predicted Species:  {predicted_species}",
            fg="#2980b9"
        )

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
    except Exception as e:
        messagebox.showerror("Runtime Error", f"Something went wrong:\n{e}")


def run_gui():
    root = tk.Tk()
    root.title("Penguin Classifier - Task 2")
    root.geometry("650x700")
    root.configure(bg="#f8f9fa")

    # ── Scrollable Frame Setup ────────────────────────────────────────────────
    main_container = tk.Frame(root, bg="#f8f9fa")
    main_container.pack(fill="both", expand=True)

    canvas = tk.Canvas(main_container, bg="#f8f9fa", highlightthickness=0)
    scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="#f8f9fa")

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Mouse wheel scrolling
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Adjust canvas width when window is resized
    def _on_canvas_configure(event):
        canvas.itemconfig(canvas.find_all()[0], width=event.width)
    canvas.bind("<Configure>", _on_canvas_configure)

    # ── Content inside scroll_frame ───────────────────────────────────────────
    
    # Title
    tk.Label(scroll_frame, text="MLP PENGUIN CLASSIFIER", font=("Arial", 16, "bold"),
             bg="#f8f9fa", pady=20).pack()

    # Network Settings
    input_frame = tk.LabelFrame(scroll_frame, text=" Network Settings ", padx=20, pady=20)
    input_frame.pack(pady=10, padx=20, fill="x")

    tk.Label(input_frame, text="Neurons (e.g. 8,4):").grid(row=0, column=0, sticky="w")
    layer_entry = ttk.Entry(input_frame); layer_entry.insert(0, "8"); layer_entry.grid(row=0, column=1, padx=10)

    tk.Label(input_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
    lr_entry = ttk.Entry(input_frame); lr_entry.insert(0, "0.1"); lr_entry.grid(row=1, column=1, padx=10)

    tk.Label(input_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
    epoch_entry = ttk.Entry(input_frame); epoch_entry.insert(0, "300"); epoch_entry.grid(row=2, column=1, padx=10)

    tk.Label(input_frame, text="Activation:").grid(row=3, column=0, sticky="w")
    act_var = tk.StringVar(value="Sigmoid")
    ttk.Combobox(input_frame, textvariable=act_var, values=["Sigmoid", "Tanh"],
                 state="readonly").grid(row=3, column=1, padx=10)

    bias_var = tk.BooleanVar(value=True)
    tk.Checkbutton(input_frame, text="Use Bias", variable=bias_var).grid(row=4, columnspan=2)

    # Objects for mapping
    # Initialized here so they are in scope for the button command
    entries  = {'layers': layer_entry, 'lr': lr_entry, 'epochs': epoch_entry,
                'act': act_var, 'bias': bias_var}

    # Train button
    tk.Button(scroll_frame, text="START TRAINING & EVALUATE", bg="#2980b9", fg="white",
              font=("Arial", 11, "bold"), pady=10,
              command=lambda: start_training_process(entries, displays)).pack(pady=10, fill="x", padx=40)

    # Results
    res_frame = tk.LabelFrame(scroll_frame, text=" Results ", padx=20, pady=20)
    res_frame.pack(pady=5, padx=20, fill="both")

    acc_l = tk.Label(res_frame, text="Accuracy: -- %", font=("Arial", 12, "bold"))
    acc_l.pack()
    m_box = tk.Text(res_frame, height=8, width=55, font=("Courier", 10))
    m_box.pack(pady=10)

    displays = {'acc_label': acc_l, 'matrix_box': m_box}

    # Classification (Single Sample)
    cls_frame = tk.LabelFrame(scroll_frame, text=" Classification (Single Sample) ", padx=20, pady=15)
    cls_frame.pack(pady=10, padx=20, fill="x")

    fields = [
        ("Culmen Length (mm):",  "culmen_length",  "45.0"),
        ("Culmen Depth (mm):",   "culmen_depth",   "15.0"),
        ("Flipper Length (mm):", "flipper_length", "200.0"),
        ("Body Mass (g):",       "body_mass",      "4000.0"),
    ]

    sample_entries = {}
    for r, (label, key, default) in enumerate(fields):
        tk.Label(cls_frame, text=label, anchor="w").grid(row=r, column=0, sticky="w", pady=3)
        e = ttk.Entry(cls_frame, width=15)
        e.insert(0, default)
        e.grid(row=r, column=1, padx=10, pady=3, sticky="w")
        sample_entries[key] = e

    # Location dropdown
    tk.Label(cls_frame, text="Location:", anchor="w").grid(row=len(fields), column=0, sticky="w", pady=3)
    loc_var = tk.StringVar(value="Biscoe")
    loc_combo = ttk.Combobox(cls_frame, textvariable=loc_var,
                             values=["Torgersen", "Biscoe", "Dream"],
                             state="readonly", width=13)
    loc_combo.grid(row=len(fields), column=1, padx=10, pady=3, sticky="w")
    sample_entries['location'] = loc_var

    # Result label
    result_lbl = tk.Label(cls_frame, text="Predicted Species:  --",
                          font=("Arial", 12, "bold"), fg="#7f8c8d", pady=5)
    result_lbl.grid(row=len(fields)+1, columnspan=2, pady=8)

    # Classify button
    tk.Button(cls_frame, text="CLASSIFY SAMPLE", bg="#27ae60", fg="white",
              font=("Arial", 10, "bold"), pady=6,
              command=lambda: classify_sample(sample_entries, result_lbl)
              ).grid(row=len(fields)+2, columnspan=2, pady=5)

    root.mainloop()


if __name__ == "__main__":
    run_gui()