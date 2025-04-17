import streamlit as st
import speech_recognition as sr
import pyttsx3
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
import numpy as np
import cv2
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class MultimodalViT(nn.Module):
    def __init__(self, num_numerical_features, num_classes, dropout_rate=0.3):
        super(MultimodalViT, self).__init__()
        
        
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Identity()  
        
        
        self.num_encoder = nn.Sequential(
            nn.Linear(num_numerical_features, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout_rate)
        )
        
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)  
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))  
        
       
        self.fc = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, numerical_features):
        
        img_features = self.vit(image)  
        img_features = img_features.unsqueeze(0)  
        
        
        num_features = self.num_encoder(numerical_features)  
        num_features = num_features.unsqueeze(0)  
        
        
        num_features = nn.functional.pad(num_features, (0, 768 - 128))  
        
       
        batch_size = image.size(0)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)  
        
        
        combined_features = torch.cat((cls_tokens, img_features, num_features), dim=0)  
        
        
        attn_output, _ = self.cross_attention(cls_tokens, combined_features, combined_features)
        
        
        output = self.fc(attn_output.squeeze(0))  
        return output

# Initialize the model
num_numerical_features = 7  
num_classes = 22  
model = MultimodalViT(num_numerical_features, num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()


model.load_state_dict(torch.load(r"C:\Users\Samridhaa\OneDrive\Desktop\New_DL\vit_multimodal_best.pth", map_location=torch.device('cpu')))
model.eval()


class_mapping = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy',
    4: 'Blueberry___healthy', 5: 'Cherry_(including_sour)___Powdery_mildew', 6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight', 10: 'Corn_(maize)___healthy', 11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)', 13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)', 16: 'Peach___Bacterial_spot', 17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot', 19: 'Pepper,_bell___healthy', 20: 'Potato___Early_blight', 21: 'Potato___healthy'
}


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
class MultimodalCropDataset(Dataset):
    def __init__(self, df, transform=None, scaler=None):
        self.df = df.copy()
        self.transform = transform
        self.scaler = scaler if scaler else StandardScaler()
        
        
        numerical_cols = self.df.columns[:-2]  
        if scaler is None:
            self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
        else:
            self.df[numerical_cols] = self.scaler.transform(self.df[numerical_cols])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["image_path"]
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        numerical_features = pd.to_numeric(self.df.iloc[idx, :-2], errors='coerce').values.astype('float32')
        numerical_features = torch.tensor(numerical_features, dtype=torch.float32)
        label = torch.tensor(self.df.iloc[idx]["Label"], dtype=torch.long)
        return image, numerical_features, label
    
    
csv_file = r"C:\Users\Samridhaa\OneDrive\Desktop\New_DL\mapped_data_with_images.csv"
df = pd.read_csv(csv_file)


label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])
num_classes = len(label_encoder.classes_)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Label"], random_state=42)
train_dataset = MultimodalCropDataset(train_df, transform=transform)
test_dataset = MultimodalCropDataset(test_df, transform=transform, scaler=train_dataset.scaler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for images, num_features, labels in test_loader:
            images, num_features, labels = images.to(device), num_features.to(device), labels.to(device)
            outputs = model(images, num_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(test_loader.dataset)
    val_acc = accuracy_score(targets, preds)
    return val_loss, val_acc, preds, targets


def predict_leaf_class(image, numerical_features=None):
    image_tensor = transform(image).unsqueeze(0)
    if numerical_features is None:
        numerical_features = torch.zeros((1, num_numerical_features))  
    with torch.no_grad():
        output = model(image_tensor, numerical_features)
        _, predicted = torch.max(output, 1)
        class_label = class_mapping[predicted.item()]
    return class_label


engine = pyttsx3.init()
engine.setProperty('rate', 150)


def speak(text):
    engine.say(text)
    engine.runAndWait()


api_key = "AIzaSyAu-weudZlsrpiyCeqD8cbKI8OPTAMWKWs"
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    google_api_key=api_key
)


def chat_with_ai(user_input):
    crop_restrict_prompt = (
        "You are a crop and plant disease assistant. "
        "Only answer questions related to crop health, leaf diseases, and remedies. "
        "If the question is unrelated, say 'Sorry, I can only help with crop-related queries.'\n"
    )
    try:
        response = chat_model.invoke(crop_restrict_prompt + user_input)
        return response.content
    except Exception as e:
        return f"Error: {e}"


def analyze_image(image_obj, question):
    crop_restrict_prompt = (
        "You are a crop disease detection assistant. "
        "Only answer crop/plant/leaf related queries. "
        "Based on this leaf image, answer: "
    )
    try:
        response = chat_model.invoke(input=[crop_restrict_prompt + question, image_obj])
        return response.content
    except Exception as e:
        return f"Error processing image: {e}"

# Streamlit UI
st.set_page_config(page_title="Crop Health Assistant", page_icon="🌿")
st.title("🌿 Crop Health Assistant")

mode = st.radio("Choose input method:", ["Text", "Voice", "Image"], horizontal=True)

if mode == "Text":
    question = st.text_input("Ask your crop-related question:")
    if question:
        response = chat_with_ai(question)
        st.success(response)
        if st.button("Speak Answer"):
            speak(response)

elif mode == "Voice":
    if st.button("Start Listening"):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Adjusting for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            st.info("Listening... Please speak.")
            try:
                audio = recognizer.listen(source, timeout=10)
                question = recognizer.recognize_google(audio)
                st.write("You said:", question)
                response = chat_with_ai(question)
                st.success(response)
                speak(response)
            except Exception as e:
                st.error(f"Error: {e}")

elif mode == "Image":
    uploaded_image = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])
    question = st.text_input("What would you like to know about this leaf?")
    if uploaded_image and question:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

        
        if any(x in question.lower() for x in ["problem", "disease", "healthy", "crop", "what is this"]):
            vit_result = predict_leaf_class(image)
            st.info(f"ViT Prediction: {vit_result}")
            final_prompt = f"This is a {vit_result}. {question}"
            response = chat_with_ai(final_prompt)
        else:
            response = analyze_image(image, question)

        st.success(response)
        if st.button("Speak Answer"):
            speak(response)
