import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
import os

# ----------------------
# Load trained model
# ----------------------
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load(r"C:\Users\anush\Desktop\machine_learning\MobileNetV2_best.pth", map_location=torch.device('cpu')))
model.eval()

# ----------------------
# Image preprocessing
# ----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# ----------------------
# Predict Function
# ----------------------
def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return img_tensor


def predict():
    if image_path.get():
        input_tensor = preprocess_image(image_path.get())
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()
            result = "Malignant (Cancer)" if predicted_class == 1 else "Benign (No Cancer)"
            messagebox.showinfo("Result", f"Prediction: {result}")
    else:
        messagebox.showerror("Error", "Please upload an image first.")


# ----------------------
# GUI Functions
# ----------------------
def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    image_path.set(file_path)
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img

# ----------------------
# Tkinter Setup
# ----------------------
root = tk.Tk()
root.title("Breast Cancer Image Classifier (PyTorch)")
root.geometry("400x400")

image_path = tk.StringVar()

tk.Button(root, text="Upload Image", command=browse_image).pack(pady=10)
image_label = tk.Label(root)
image_label.pack()

tk.Button(root, text="Predict", command=predict).pack(pady=10)

root.mainloop()
