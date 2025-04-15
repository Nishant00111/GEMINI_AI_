# from dotenv import load_dotenv
# import os
# from google.generativeai import configure, list_models

# # Load the .env file
# load_dotenv()

# # Get the API key from the environment
# api_key = os.getenv("GOOGLE_API_KEY")

# if api_key:
#     # Pass the API key to the Google Gemini API
#     configure(api_key=api_key)
#     print("API Key loaded successfully!")
# else:
#     print("API Key not found in .env file!")

# # Get the list of models
# models = list_models()

# # Iterate and print model details
# for model in models:
#     # Access model attributes using dot notation
#     print(f"Model Name: {model.name}")  # Assuming 'name' is an attribute of the model
#     print(f"Model Description: {model.description}")  # Assuming 'description' is an attribute
#     print("------")



# IMAGE GEN APP ---->

import streamlit as st
from diffusers import DiffusionPipeline
import torch

# Function to load the model
@st.cache_resource
def load_models():
    # Load base model
    base_model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    base_model.to("cuda")

    # Load refiner model
    refiner_model = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base_model.text_encoder_2,
        vae=base_model.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    )
    refiner_model.to("cuda")

    return base_model, refiner_model

# Function to generate images
def generate_image(prompt, base_model, refiner_model, n_steps=40, high_noise_frac=0.8):
    # Run base model to generate latent image
    image = base_model(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    
    # Use refiner model to refine the image
    image = refiner_model(
        prompt=prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    
    return image

# Streamlit UI components
st.title("Stable Diffusion XL Image Generator")
st.write("Generate images using Stable Diffusion XL models with and without refinement.")

# Prompt input
prompt = st.text_input("Enter your prompt:", "A majestic lion jumping from a big stone at night")

# Load models when the app starts
base_model, refiner_model = load_models()

# Button to trigger image generation
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            generated_image = generate_image(prompt, base_model, refiner_model)
            st.image(generated_image, caption="Generated Image", use_column_width=True)
    else:
        st.error("Please enter a prompt!")

# Allow user to adjust number of steps
steps = st.slider("Number of Inference Steps", min_value=10, max_value=100, value=40)

# Option to enable/disable refinement
use_refinement = st.checkbox("Use Refiner (Higher quality)", value=True)

if use_refinement:
    st.write("Using the base+refiner pipeline for higher quality.")
else:
    st.write("Using the base model only.")
