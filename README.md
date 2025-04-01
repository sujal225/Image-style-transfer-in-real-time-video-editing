# AI-Based Image Style Transfer

This project implements real-time artistic style transfer using Generative Adversarial Networks (GANs) and compiler strategies to optimize artistic style transfer for videos and images. The model is based on a Transformer network and a pre-trained VGG model for feature extraction.

## Features

- Real-time image and video style transfer
- Training custom styles using your dataset
- GUI-based execution using Tkinter
- Optimized model using compiler strategies

---

## Installation

### **Prerequisites**

Make sure you have the following installed:

- Python 3.10+
- PyTorch
- torchvision
- NumPy
- OpenCV
- Tkinter (for GUI)

### **Setup Instructions**

```bash
# Clone the repository
git clone https://github.com/yourusername/style-transfer.git
cd style-transfer
```
```bash
# Install dependencies
pip install torch torchvision numpy pillow tqdm opencv-python  
```

---

## Usage

### **Run Pre-trained Model**

```bash
python style_transfer.py eval --content content.jpg --model trained_models/model.pth --output output.jpg
```

### **Train a New Model**

```bash
python style_transfer.py train --dataset path/to/dataset --epochs 5 --style-image style.jpg --output-model trained_models/new_model.pth
```

#### **GUI Mode**

Run the Tkinter-based GUI:

- video_st_inputbox.py [To use pre-traiined model]
- train.py [To train a model]

---

## Training

### **Dataset Structure**

Make sure your dataset is structured as follows:

```
/dataset_name
  /train
    /class1
      img1.jpg
      img2.jpg
    /class2
      img3.jpg
      img4.jpg
  /test
    /class1
      img5.jpg
    /class2
      img6.jpg
```


---

## Troubleshooting

### **FileNotFoundError: Couldn't find any class folder in dataset path**

**Solution:** Make sure the dataset is correctly placed and structured.

### **RuntimeError: Error(s) in loading state\_dict**

**Solution:** Ensure the model is saved and loaded with the same architecture.

---

## Contributors

- **Sujal Bansal** - *Overall*

---



