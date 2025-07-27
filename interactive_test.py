

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms as transforms
import numpy as np
import os

# --- Model Definition ---
# Assuming target_cnn.py is in the same directory
try:
    from target_cnn import TargetCNN
except ImportError:
    messagebox.showerror("Error", "Could not import TargetCNN. Make sure target_cnn.py is in the same directory.")
    exit()

class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SynthNet vs. Trained CNN Digit Recognizer")

        # --- Model and Device Setup ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = self.initialize_model()
        self.generated_model = self.initialize_model()
        self.transform = self.get_data_transform()

        # --- State Variables ---
        self.view_mode = tk.StringVar(value="side_by_side")
        self.trained_model_path = tk.StringVar(value="N/A")
        self.generated_model_path = tk.StringVar(value="N/A")

        # --- UI Structure ---
        self.root.configure(bg='#2E2E2E')
        self.control_frame = tk.Frame(root, bg='#3C3C3C', padx=10, pady=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.main_frame = tk.Frame(root, bg='#2E2E2E', padx=10, pady=10)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.create_controls()
        self.setup_views()
        
        # Load initial models
        self.load_initial_weights()

    def initialize_model(self):
        """Creates an instance of the TargetCNN and moves it to the correct device."""
        try:
            model = TargetCNN().to(self.device)
            model.eval()
            return model
        except Exception as e:
            messagebox.showerror("Model Initialization Error", f"Failed to initialize TargetCNN: {e}")
            self.root.destroy()

    def get_data_transform(self):
        """Returns the same transform used for MNIST training."""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def create_controls(self):
        """Creates the control panel widgets at the top of the GUI."""
        # View Mode Radio Buttons
        tk.Label(self.control_frame, text="View Mode:", bg='#3C3C3C', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(self.control_frame, text="Side-by-Side", variable=self.view_mode, value="side_by_side", command=self.setup_views, bg='#3C3C3C', fg='white', selectcolor='#555555').pack(side=tk.LEFT)
        tk.Radiobutton(self.control_frame, text="Toggle", variable=self.view_mode, value="toggle", command=self.setup_views, bg='#3C3C3C', fg='white', selectcolor='#555555').pack(side=tk.LEFT, padx=5)

    def setup_views(self):
        """Sets up the main view based on the selected mode."""
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        if self.view_mode.get() == "side_by_side":
            self.create_side_by_side_view()
        else:
            self.create_toggle_view()

    def create_side_by_side_view(self):
        """Creates two drawing panels for direct comparison."""
        self.frame1 = self.create_drawing_panel(self.main_frame, "Trained Model", self.load_trained_model, self.trained_model_path)
        self.frame2 = self.create_drawing_panel(self.main_frame, "SynthNet Generated Model", self.load_generated_model, self.generated_model_path)
        self.frame1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.frame2.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Shared action buttons
        action_frame = tk.Frame(self.main_frame, bg='#2E2E2E')
        action_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Create a single canvas for drawing
        self.shared_canvas = tk.Canvas(action_frame, width=280, height=280, bg='white', cursor="cross")
        self.shared_canvas.pack(pady=10, padx=10)
        self.shared_canvas.bind("<B1-Motion>", self.paint_shared)
        
        self.shared_pil_image = Image.new('L', (280, 280), 'black')
        self.shared_pil_draw = ImageDraw.Draw(self.shared_pil_image)

        tk.Button(action_frame, text="Predict Both", command=self.predict_both, bg='#4CAF50', fg='white', width=15).pack(side=tk.LEFT, expand=True, padx=5)
        tk.Button(action_frame, text="Clear", command=self.clear_both, bg='#F44336', fg='white', width=15).pack(side=tk.RIGHT, expand=True, padx=5)


    def create_toggle_view(self):
        """Creates a single drawing panel with a toggle for the model."""
        self.single_frame = self.create_drawing_panel(self.main_frame, "Interactive Test", None, None, is_toggle=True)
        self.single_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add model selection and loading to the single panel's control frame
        toggle_control_frame = self.single_frame.winfo_children()[0] # The control sub-frame
        
        self.active_model_var = tk.StringVar(value="trained")
        tk.Radiobutton(toggle_control_frame, text="Use Trained", variable=self.active_model_var, value="trained", bg='#3C3C3C', fg='white', selectcolor='#555555').pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(toggle_control_frame, text="Use Generated", variable=self.active_model_var, value="generated", bg='#3C3C3C', fg='white', selectcolor='#555555').pack(side=tk.LEFT, padx=5)

        tk.Button(toggle_control_frame, text="Load Trained Weights", command=self.load_trained_model, bg='#008CBA', fg='white').pack(side=tk.LEFT, padx=5)
        tk.Button(toggle_control_frame, text="Load Generated Weights", command=self.load_generated_model, bg='#008CBA', fg='white').pack(side=tk.LEFT, padx=5)
        
        tk.Label(toggle_control_frame, textvariable=self.trained_model_path, bg='#3C3C3C', fg='#CCCCCC').pack(side=tk.LEFT, padx=10)
        tk.Label(toggle_control_frame, textvariable=self.generated_model_path, bg='#3C3C3C', fg='#CCCCCC').pack(side=tk.LEFT, padx=10)
        
        action_frame = tk.Frame(self.single_frame, bg='#3C3C3C')
        action_frame.pack(fill=tk.X, pady=5)
        tk.Button(action_frame, text="Predict", command=self.predict_toggle, bg='#4CAF50', fg='white').pack(side=tk.LEFT, expand=True, padx=5)
        tk.Button(action_frame, text="Clear", command=self.clear_toggle, bg='#F44336', fg='white').pack(side=tk.RIGHT, expand=True, padx=5)


    def create_drawing_panel(self, parent, title, load_command, path_var, is_toggle=False):
        """Helper to create a self-contained drawing and prediction panel."""
        panel_frame = tk.Frame(parent, bg='#3C3C3C', bd=2, relief=tk.RIDGE)
        
        control_sub_frame = tk.Frame(panel_frame, bg='#3C3C3C', pady=5)
        control_sub_frame.pack(side=tk.TOP, fill=tk.X)
        tk.Label(control_sub_frame, text=title, font=('Arial', 16, 'bold'), bg='#3C3C3C', fg='white').pack(side=tk.LEFT, padx=10)
        
        if load_command:
            tk.Button(control_sub_frame, text="Load Weights", command=load_command, bg='#008CBA', fg='white').pack(side=tk.RIGHT, padx=10)
            tk.Label(control_sub_frame, textvariable=path_var, bg='#3C3C3C', fg='#CCCCCC').pack(side=tk.RIGHT, padx=10)

        if is_toggle:
            canvas = tk.Canvas(panel_frame, width=280, height=280, bg='white', cursor="cross")
            canvas.pack(pady=10, padx=10)
            canvas.bind("<B1-Motion>", lambda event, c=canvas: self.paint(event, c))
            panel_frame.canvas = canvas
            
            img = Image.new('L', (280, 280), 'black')
            draw = ImageDraw.Draw(img)
            panel_frame.pil_image = img
            panel_frame.pil_draw = draw

        pred_label = tk.Label(panel_frame, text="Prediction: ?", font=('Arial', 20), bg='#3C3C3C', fg='white')
        pred_label.pack(pady=10)
        panel_frame.pred_label = pred_label

        return panel_frame

    def load_initial_weights(self):
        if os.path.exists("checkpoints_weights_cnn/weights_epoch_final.pth"):
            self.load_model_weights(self.trained_model, self.trained_model_path, "checkpoints_weights_cnn/weights_epoch_final.pth")
        if os.path.exists("generalized_checkpoints_weights/weights_step_11.pth"):
            self.load_model_weights(self.generated_model, self.generated_model_path, "generalized_checkpoints_weights/weights_step_11.pth")

    def load_trained_model(self):
        path = filedialog.askopenfilename(initialdir="checkpoints_weights_cnn", title="Select Trained Weights", filetypes=(("PyTorch Models", "*.pth"),))
        if path: self.load_model_weights(self.trained_model, self.trained_model_path, path)

    def load_generated_model(self):
        path = filedialog.askopenfilename(initialdir="generalized_checkpoints_weights", title="Select Generated Weights", filetypes=(("PyTorch Models", "*.pth"),))
        if path: self.load_model_weights(self.generated_model, self.generated_model_path, path)

    def load_model_weights(self, model, path_var, file_path):
        try:
            model.load_state_dict(torch.load(file_path, map_location=self.device))
            model.eval()
            path_var.set(os.path.basename(file_path))
        except Exception as e:
            messagebox.showerror("Error loading model", f"Could not load weights from {file_path}.\nError: {e}")

    def paint_shared(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        self.shared_canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.shared_pil_draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def paint(self, event, canvas):
        panel = canvas.master
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10)
        canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        panel.pil_draw.ellipse([x1, y1, x2, y2], fill='white', outline='white')

    def preprocess_image(self, pil_image):
        img = ImageOps.invert(pil_image)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        return self.transform(img).unsqueeze(0).to(self.device)

    def predict_both(self):
        img_tensor = self.preprocess_image(self.shared_pil_image)
        with torch.no_grad():
            output1 = self.trained_model(img_tensor)
            pred1 = output1.argmax(dim=1).item()
            self.frame1.pred_label.config(text=f"Prediction: {pred1}")
            
            output2 = self.generated_model(img_tensor)
            pred2 = output2.argmax(dim=1).item()
            self.frame2.pred_label.config(text=f"Prediction: {pred2}")

    def predict_toggle(self):
        img_tensor = self.preprocess_image(self.single_frame.pil_image)
        model = self.trained_model if self.active_model_var.get() == "trained" else self.generated_model
        with torch.no_grad():
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            self.single_frame.pred_label.config(text=f"Prediction: {pred} ({self.active_model_var.get()})")

    def clear_both(self):
        self.shared_canvas.delete("all")
        self.shared_pil_draw.rectangle([0, 0, 280, 280], fill='black')
        self.frame1.pred_label.config(text="Prediction: ?")
        self.frame2.pred_label.config(text="Prediction: ?")

    def clear_toggle(self):
        panel = self.single_frame
        panel.canvas.delete("all")
        panel.pil_draw.rectangle([0, 0, 280, 280], fill='black')
        panel.pred_label.config(text="Prediction: ?")

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()

