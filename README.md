# Tomato-Leaf-Disease-Classification-Using-VGG19

## 📌 Project Overview
This project applies **deep learning** techniques to **classify tomato leaf diseases** using a **pretrained VGG19 model**. The model leverages **transfer learning** to distinguish between **healthy and diseased leaves**, aiding early disease detection in tomato plants.

## 🦠 Disease Classes
The dataset contains **11 classes** of tomato leaf diseases along with healthy leaves:
1. **Bacterial Spot**
2. **Early Blight**
3. **Late Blight**
4. **Leaf Mold**
5. **Septoria Leaf Spot**
6. **Spider Mites (Two-Spotted Spider Mite)**
7. **Target Spot**
8. **Tomato Yellow Leaf Curl Virus**
9. **Tomato Mosaic Virus**
10. **Powdery Mildew**
11. **Healthy Leaves**

The model classifies an input image into one of these 11 categories.

## 🔹 Key Features
- **Pretrained VGG19 model** for feature extraction  
- **Fine-tuned classifier** for multi-class tomato leaf disease identification  
- **Dataset preprocessing**, including augmentation and normalization  
- **Evaluation metrics** such as accuracy and loss tracking  
- **Jupyter Notebook-based implementation** for easy experimentation  
Each folder contains images labeled according to the disease category.

## ⚙️ Code Explanation

### 1️⃣ Data Preprocessing
- **Image Augmentation**: Resize, normalize, and apply transformations such as rotation, flipping, and zooming to enhance model generalization.  
- **Dataset Loading**: Images are loaded using TensorFlow/Keras and converted into tensors for model training.  

### 2️⃣ Model Architecture - VGG19
- **Pretrained Weights**: The model loads VGG19 pretrained on ImageNet.  
- **Modified Classifier**: The final dense layers are replaced with custom **fully connected layers** to classify 11 disease classes.  
- **Activation Function**: Uses **Softmax** for multi-class classification.  
- **Optimizer**: Adam optimizer for better convergence.  

### 3️⃣ Training & Validation
- **Loss Function**: Categorical Cross-Entropy  
- **Evaluation Metrics**: Accuracy, Precision, Recall  
- **Early Stopping**: Prevents overfitting by monitoring validation loss.  

### 4️⃣ Prediction & Testing
- **Model Testing**: Evaluates on unseen test images.  
- **Visualization**: Displays sample predictions with confidence scores.  

## 🎯 Expected Outcome
- **High classification accuracy** across all 11 classes. 
- **Potential for real-time disease detection using edge devices.**  
