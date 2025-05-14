🧩 Problem Statement
Crop diseases threaten agricultural productivity and often lead to economic losses. Traditional detection methods are slow, labor-intensive, and heavily reliant on expert knowledge, often ignoring critical environmental conditions.
This project proposes an intelligent, scalable system using Vision Transformers (ViT) and environmental features for multimodal disease prediction, aiding timely decision-making in precision agriculture.

📄 Abstract
This project presents a Multimodal Deep Learning Framework combining:

🌿 Leaf Image Analysis using Vision Transformer (ViT)
🌡️ Environmental Sensor Data (e.g., temperature, humidity, soil moisture)
A cross-attention mechanism fuses both modalities, enhancing classification performance.
This intelligent and scalable system helps farmers detect crop diseases early and accurately, advancing smart agriculture.

📊 Dataset Description
The dataset is sourced from Kaggle:
🔗 Multimodal Plant Disease Dataset by Subham Divakar

Each sample includes:

Leaf Images: High-resolution RGB images labeled with disease categories.
Numerical Features: 7 tabular features (e.g., temperature, humidity, pH, rainfall).
All data are stored in mapped_data_with_images.csv, with rows linking image paths and environmental attributes.
Dataset split: 80% training, 20% testing (stratified sampling).

🛠️ Tools and Technologies
Development Environment
Python 3.x
Jupyter Notebook / VS Code
Google Colab
Git & GitHub
Streamlit (web app)
Google Gemini (via LangChain)
Key Libraries
Category	Libraries
Data	pandas, numpy, scikit-learn
Visualization	matplotlib, seaborn
Computer Vision	OpenCV, PIL
Deep Learning	PyTorch, torchvision, EfficientNet, ResNet50, MobileNetV2, ViT
Voice/Chat AI	speech_recognition, pyttsx3, langchain_google_genai
🧠 Methods & Implementation
1. Preprocessing
Images: Resized (224×224), augmented (flip, rotation), normalized
Numerical Data: Standard Scaler used; class labels encoded
2. Model Architecture
Multimodal Neural Network
Visual Branch: ViT-B_16 (ImageNet pretrained)
Numerical Branch: Feedforward NN (Linear → ReLU → LayerNorm → Dropout)
Fusion: Multihead Cross-Attention
Classifier: MLP with final softmax output
3. Training
Loss: CrossEntropyLoss
Optimizer: Adam (lr=0.0001)
Epochs: 15 (early stopping)
Metrics: Accuracy, F1-score, Confusion Matrix
4. Inference
Model file: vit_multimodal_best.pth
Inputs: Leaf image + optional numerical features
Outputs: Disease class + confidence score
5. Streamlit Web Interface
🔼 Upload images
📈 Enter optional environmental features
🎙️ Voice or 💬 text interaction with Gemini-powered chatbot
🔊 Text-to-speech support via pyttsx3
📈 Model Evaluation
Metric	Value
Test Accuracy	97.95%
F1-Score	0.94
Precision	High (most > 90%)
Recall	Balanced across classes

![image](https://github.com/user-attachments/assets/09332cd4-f4a3-41c0-b016-bd8bdebf2e6b)


The above image showcases the homepage interface of the Crop Health Assistant, a user-friendly web application designed to facilitate easy access to crop disease diagnosis. The interface allows users to select from three input methods—Text, Voice, and Image—to interact with the system, making it adaptable to various user preferences and environmental situations.

● The Text option enables users to manually enter crop-related queries.

● The Voice option supports verbal interactions, useful in hands-free or field conditions.

● The Image option allows users to upload leaf images for visual disease diagnosis using the Vision Transformer (ViT) model.

This clean, minimalistic design ensures that farmers, agricultural officers, and researchers can efficiently navigate the platform and access AI-driven crop health insights. The input box invites users to ask a question about crop health, which then triggers the multimodal model pipeline for prediction and recommendation.

![image](https://github.com/user-attachments/assets/21e57ee9-f600-4cd3-abbb-033ba82d1718)
![443776495-f29f6e34-8fa7-44c6-8b3d-b6057ed15cb3](https://github.com/user-attachments/assets/3e36c378-677a-4677-9960-12c35f239b4e)
![image](https://github.com/user-attachments/assets/4c76b72f-37fb-4ecd-b66c-fdd652da071b)


Upon uploading a leaf image, the system utilizes a Vision Transformer (ViT) model to analyze visual features and predict the crop disease. In this instance, the model identified the disease as:

ViT Prediction: Orange_Huanglongbing (Citrus greening) with 99.06% confidence

Following the prediction, the user posed a natural-language follow-up question: "What diseases can affect this?"

The system responds with contextual information, explaining that:

● Citrus greening (Huanglongbing or HLB) is the primary disease.

● While not directly caused by other diseases, it weakens the tree.

● This makes the plant more prone to secondary infections from fungi and pathogens.

● These secondary infections can accelerate the decline of the tree’s health.

![image](https://github.com/user-attachments/assets/4aee1fdc-8616-417b-8705-6a8bf113cce3)

