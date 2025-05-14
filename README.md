# 🌾 Crop Health Assistant

## OVERVIEW

**Crop Health Assistant** is a multimodal AI system for early and accurate crop disease detection. It integrates:

- A **Vision Transformer (ViT)** for leaf image classification  
- **Environmental sensor features** (e.g., temperature, humidity, soil moisture)  
- A **cross-attention fusion model** for combining image and numerical data  
- A user-friendly **Streamlit web interface** with **voice, text, and image input** support  
- **Google Gemini** integration for natural language Q&A on crop health

This system empowers farmers and agricultural professionals with real-time, intelligent insights for precision agriculture.

---

## 🎯 FEATURES

- Diagnose plant diseases using:
  - Leaf **image uploads**
  - **Environmental data** input (optional)
  - Natural **text or voice queries**
- Multimodal disease prediction using ViT + sensor data
- Google Gemini chatbot for contextual crop-related questions
- Voice output via text-to-speech (pyttsx3)
- Clean, intuitive UI adaptable for mobile/field conditions

---

## 🧠 MODEL ARCHITECTURE

### 🔍 Components

| Component         | Description                                               |
|------------------|-----------------------------------------------------------|
| **Image Encoder** | ViT-B_16 pretrained on ImageNet                           |
| **Tabular Branch**| Feedforward NN (Linear → ReLU → LayerNorm → Dropout)     |
| **Fusion**        | Multihead Cross-Attention to merge image and tabular data|
| **Classifier**    | Final MLP + Softmax                                       |

### 🧪 Training Setup

- **Loss**: CrossEntropyLoss  
- **Optimizer**: Adam (lr=0.0001)  
- **Epochs**: 15 (with Early Stopping)  
- **Metrics**: Accuracy, F1-score, Confusion Matrix  

---

## 📊 DATASET

- 🔗 **Multimodal Plant Disease Dataset by Subham Divakar (Kaggle)**
- 80-20 stratified split (Training / Testing)
- **Leaf Images**: RGB, high-resolution, labeled
- **Numerical Data**: 7 features (e.g., temperature, humidity, pH, rainfall)
- Dataset reference file: `mapped_data_with_images.csv`

---

## 🛠 TECHNOLOGIES USED

### Backend & Development

- **Python 3.x**, **Jupyter**, **VS Code**, **Google Colab**
- **Streamlit** – Web interface
- **Git & GitHub** – Version control
- **LangChain + Gemini** – Q&A chatbot

### Libraries

| Category              | Libraries Used                                                                 |
|-----------------------|--------------------------------------------------------------------------------|
| Data Handling         | `pandas`, `numpy`, `scikit-learn`                                              |
| Visualization         | `matplotlib`, `seaborn`                                                        |
| Computer Vision       | `OpenCV`, `PIL`, `torchvision`, `torch`, `ViT`, `EfficientNet`, `MobileNetV2` |
| Voice/Chat AI         | `speech_recognition`, `pyttsx3`, `langchain_google_genai`                     |

---

## 🚀 INFERENCE PIPELINE

- **Input**: Leaf image + optional environmental values  
- **Output**: Disease class with confidence  
- **Model file**: `vit_multimodal_best.pth`  
- Gemini chatbot enables follow-up Q&A like:
  - "What is this disease?"
  - "How to treat it?"
  - "What crops are vulnerable?"

---

## ✅ MODEL PERFORMANCE

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | **97.95%**|
| F1-Score     | **0.94**  |
| Precision    | High (>90%) for most classes |
| Recall       | Balanced across all disease categories |

---

## 💻 WEB APP SNAPSHOTS

![Homepage](https://github.com/user-attachments/assets/09332cd4-f4a3-41c0-b016-bd8bdebf2e6b)  
*Select from Text, Voice, or Image for interaction*

---

![ViT Prediction](https://github.com/user-attachments/assets/4c76b72f-37fb-4ecd-b66c-fdd652da071b)  
*Image-based prediction: Citrus Greening (99.06% confidence)*

---

![Chat Interaction](https://github.com/user-attachments/assets/4aee1fdc-8616-417b-8705-6a8bf113cce3)  
*Gemini answers follow-up crop health queries*

---

## 🔗 FUTURE WORK

- Add geospatial data support (e.g., GPS coordinates)
- Real-time sensor integration via IoT
- Disease progression tracking
- Multilingual support for farmers in regional languages
