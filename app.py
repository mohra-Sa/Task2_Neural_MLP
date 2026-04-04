import tkinter as tk
from tkinter import ttk, messagebox

from preprocessing import preprocess_data
from mlp import MultiLayerPerceptron
from matrices import compute_confusion_matrix, compute_accuracy

def start_training_process(entries, display_widgets):
    try:
    
        lr = float(entries['lr'].get())
        epochs = int(entries['epochs'].get())
        act_type = entries['act'].get()
        use_bias = entries['bias'].get()
        
        h_layers = [int(n.strip()) for n in entries['layers'].get().split(",")]

        X_train, y_train, X_test, y_test, species_list = preprocess_data()
      
        mlp = MultiLayerPerceptron(
            input_size=5, 
            hidden_layers=h_layers, 
            output_size=3, 
            learning_rate=lr, 
            activation_type=act_type, 
            use_bias=use_bias
        )

        
        mlp.train(X_train, y_train, epochs=epochs)
        y_pred = mlp.test(X_test) # استقبال الأوتبوت

      
        acc = compute_accuracy(y_test, y_pred)
        matrix = compute_confusion_matrix(y_test, y_pred)

        
        display_widgets['acc_label'].config(text=f"Accuracy: {acc:.2f} %", fg="#27ae60")
        
        # confusion matrix display
        display_widgets['matrix_box'].delete("1.0", tk.END)
        header = f"{'Actual/Pred':<12} | " + " | ".join([f"{s[:6]:>8}" for s in species_list]) + "\n"
        display_widgets['matrix_box'].insert(tk.END, header + "-"*45 + "\n")
        
        for i, row in enumerate(matrix):
            row_str = f"{species_list[i][:10]:<12} | " + " | ".join([f"{val:>8}" for val in row]) + "\n"
            display_widgets['matrix_box'].insert(tk.END, row_str)

    except Exception as e:
        messagebox.showerror("Runtime Error", f"Something went wrong:\n{e}")

def run_gui():
    root = tk.Tk()
    root.title("Penguin Classifier - Task 2")
    root.geometry("600x800")
    root.configure(bg="#f8f9fa")

    #title
    tk.Label(root, text="MLP PENGUIN CLASSIFIER", font=("Arial", 16, "bold"), bg="#f8f9fa", pady=20).pack()

    #settings frame
    input_frame = tk.LabelFrame(root, text=" Network Settings ", padx=20, pady=20)
    input_frame.pack(pady=10, padx=20, fill="x")

    tk.Label(input_frame, text="Neurons (e.g. 8,4):").grid(row=0, column=0, sticky="w")
    layer_entry = ttk.Entry(input_frame); layer_entry.insert(0, "8"); layer_entry.grid(row=0, column=1, padx=10)

    tk.Label(input_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
    lr_entry = ttk.Entry(input_frame); lr_entry.insert(0, "0.1"); lr_entry.grid(row=1, column=1, padx=10)

    tk.Label(input_frame, text="Epochs:").grid(row=2, column=0, sticky="w")
    epoch_entry = ttk.Entry(input_frame); epoch_entry.insert(0, "1000"); epoch_entry.grid(row=2, column=1, padx=10)

    tk.Label(input_frame, text="Activation:").grid(row=3, column=0, sticky="w")
    act_var = tk.StringVar(value="Sigmoid")
    ttk.Combobox(input_frame, textvariable=act_var, values=["Sigmoid", "Tanh"], state="readonly").grid(row=3, column=1, padx=10)

    bias_var = tk.BooleanVar(value=True)
    tk.Checkbutton(input_frame, text="Use Bias", variable=bias_var).grid(row=4, columnspan=2)

     #button to start training
    tk.Button(root, text="START TRAINING & EVALUATE", bg="#2980b9", fg="white", 
              font=("Arial", 11, "bold"), pady=10,
              command=lambda: start_training_process(entries, displays)).pack(pady=20, fill="x", padx=40)

    #results frame
    res_frame = tk.LabelFrame(root, text=" Results ", padx=20, pady=20)
    res_frame.pack(pady=10, padx=20, fill="both", expand=True)

    acc_l = tk.Label(res_frame, text="Accuracy: -- %", font=("Arial", 12, "bold"))
    acc_l.pack()
    m_box = tk.Text(res_frame, height=8, width=50, font=("Courier", 10))
    m_box.pack(pady=10)

    entries = {'layers': layer_entry, 'lr': lr_entry, 'epochs': epoch_entry, 'act': act_var, 'bias': bias_var}
    displays = {'acc_label': acc_l, 'matrix_box': m_box}

   

    root.mainloop()

if __name__ == "__main__":
    run_gui()