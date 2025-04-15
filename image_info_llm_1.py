# from dotenv import load_dotenv
# import streamlit as st
# import os
# from PIL import Image
# import io
# import google.generativeai as genai

# # Load environment variables
# load_dotenv()

# # Configure Gemini API
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Function to process images and get response
# def get_gemini_response(input_text, image):
#     model = genai.GenerativeModel('gemini-1.5-pro')  # Use the best model
    
#     image_data = None
#     if image:
#         # Convert image to bytes for Gemini API
#         img_byte_array = io.BytesIO()
#         image.save(img_byte_array, format="PNG")
#         img_byte_array = img_byte_array.getvalue()
#         image_data = {"mime_type": "image/png", "data": img_byte_array}
    
#     if input_text:
#         response = model.generate_content([input_text, image_data])
#     else:
#         response = model.generate_content(image_data)
    
#     return response.text

# # Initialize Streamlit app
# st.set_page_config(page_title="Gemini Image Demo")
# st.header("Gemini Application")

# # User input prompt
# input_text = st.text_input("Input Prompt: ", key="input")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
# image = None
# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image.", use_column_width=True)

# # Process request
# if st.button("Tell me about the image"):
#     if not uploaded_file:
#         st.error("Please upload an image first.")
#     else:
#         response = get_gemini_response(input_text, image)
#         st.subheader("The Response is:")
#         st.write(response)





from dotenv import load_dotenv
import streamlit as st
import os
import io
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load DINOv2 model from Hugging Face
device = "cuda" if torch.cuda.is_available() else "cpu"
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)

# Image Feature Extraction
def extract_features(image):
    """Extract meaningful embeddings using DINOv2."""
    image = dino_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = dino_model(**image).last_hidden_state.mean(dim=1)  # Get pooled features
    return features.cpu().numpy().tolist()

# Function to interact with Gemini API
def chat_with_gemini(chat_history, input_text, image):
    """Chat with Gemini about the uploaded image."""
    model = genai.GenerativeModel('gemini-1.5-pro')

    # Extract image features
    image_features = extract_features(image)

    # Convert image to bytes
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format="PNG")
    img_byte_array = img_byte_array.getvalue()
    
    image_data = {"mime_type": "image/png", "data": img_byte_array}

    # Structured prompt for deep analysis
    prompt = f"""
    You are an expert AI trained to analyze images with deep insights. 
    The user has uploaded an image and will ask questions about it. 

    - Identify objects, colors, and patterns.
    - Provide detailed analysis of textures and materials.
    - If text is present, transcribe and analyze it.
    - Compare image features with known patterns or styles.
    - If asked about emotions, describe the mood conveyed by the image.
    - If technical details are required, analyze lighting, angles, and composition.

    Always provide clear and concise responses.
    """

    # Prepare conversation history
    chat_input = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Here is the uploaded image."},
        {"role": "model", "content": str(image_features)},
    ]

    # Add user conversation history
    for message in chat_history:
        chat_input.append({"role": "user", "content": message["user"]})
        chat_input.append({"role": "model", "content": message["model"]})

    # Append the latest user question
    chat_input.append({"role": "user", "content": input_text})

    # Generate response
    response = model.generate_content([prompt, input_text, image_data, str(image_features)])
    
    return response.text

# Streamlit App
st.set_page_config(page_title="Chat with AI about an Image")
st.header("Gemini AI - Image Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Chat input
input_text = st.text_input("Ask about the image: ")

# Chat button
if st.button("Ask AI"):
    if not uploaded_file:
        st.error("Please upload an image first.")
    elif not input_text:
        st.error("Please enter a question.")
    else:
        response = chat_with_gemini(st.session_state.chat_history, input_text, image)
        
        # Store conversation history
        st.session_state.chat_history.append({"user": input_text, "model": response})

# Display chat history
st.subheader("Chat History")
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**AI:** {chat['model']}")
    st.write("---")



