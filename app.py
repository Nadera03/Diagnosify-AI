import gradio as gr
import torch
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image

# Load models only once for speed
emotion_model = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=0 if torch.cuda.is_available() else -1)
gut_health_model = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0 if torch.cuda.is_available() else -1)
retina_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
retina_model.eval()

# Function: Emotion-based Disease Detection
def detect_disease_from_emotion(text):
    emotions = emotion_model(text)
    emotion_label = emotions[0]['label']
    disease_mapping = {
        "anger": "High blood pressure, Heart Disease",
        "joy": "Generally healthy",
        "sadness": "Depression, Low Immunity",
        "fear": "Anxiety Disorders",
        "surprise": "No major risks"
    }
    return disease_mapping.get(emotion_label, "No specific disease found.")

# Function: Gut Health Analysis
def analyze_gut_health(diet_input):
    result = gut_health_model(diet_input)
    age_range = result[0]['label']
    return f"Your gut microbiome resembles a person in the {age_range} age range."

# Function: Retina Scan Disease Detection
def detect_disease_from_retina(image):
    image = Image.open(image).convert("RGB")
    image = image.resize((224, 224))
    img_tensor = torch.tensor(np.array(image)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        output = retina_model(img_tensor)
    return f"Retina analysis complete. Model confidence score: {output.max().item():.2f}"

# UI Design
with gr.Blocks() as app:
    gr.Markdown("# 🏥 Diagnosify-AI: Your AI Health Assistant")
    with gr.Tab("🧠 Emotion-to-Disease"):
        gr.Markdown("Enter your emotions, and Diagnosify-AI will predict possible health risks.")
        emotion_input = gr.Textbox(label="Describe your current feelings")
        emotion_output = gr.Textbox(label="Possible Health Risks")
        gr.Button("Analyze").click(detect_disease_from_emotion, inputs=emotion_input, outputs=emotion_output)

    with gr.Tab("🍽️ Gut Health Analysis"):
        gr.Markdown("Enter your daily diet to analyze your gut health status.")
        diet_input = gr.Textbox(label="Describe your daily food intake")
        gut_output = gr.Textbox(label="Gut Health Insights")
        gr.Button("Analyze").click(analyze_gut_health, inputs=diet_input, outputs=gut_output)

    with gr.Tab("👁️ Retina Disease Scan"):
        gr.Markdown("Upload an image of your retina for disease analysis.")
        retina_input = gr.Image(type="filepath")  # or "numpy" if needed
        retina_output = gr.Textbox(label="Analysis Result")
        gr.Button("Scan").click(detect_disease_from_retina, inputs=retina_input, outputs=retina_output)

# Launch App
app.launch()


