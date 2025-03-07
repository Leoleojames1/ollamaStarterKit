import gradio as gr
import requests
import os
import time
from PIL import Image
import io
import base64
import ollama
import uuid

def enhance_prompt_with_ollama(prompt, ollama_model):
    """Use Ollama to enhance the provided prompt"""
    try:
        messages = [{'role': 'user', 'content': f"Enhance this image generation prompt to make it more detailed and visually appealing. Only return the enhanced prompt, no explanations: '{prompt}'"}]
        response = ollama.chat(model=ollama_model, messages=messages)
        enhanced_prompt = response['message']['content'].strip()
        return enhanced_prompt
    except Exception as e:
        print(f"Error enhancing prompt with Ollama: {e}")
        return prompt  # Return original prompt if enhancement fails

def generate_images(prompt, api_key, model_name, batch_count, use_prompt_boost, ollama_model):
    """Generate images using the provided parameters"""
    results = []
    error_messages = []
    
    # Validate inputs
    if not prompt.strip():
        return None, "Please provide a prompt."
    if not api_key.strip():
        return None, "Please provide an API key."
    
    try:
        batch_count = int(batch_count)
        if batch_count < 1:
            batch_count = 1
    except:
        batch_count = 1
    
    for i in range(batch_count):
        current_prompt = prompt
        
        # Enhance prompt if requested
        if use_prompt_boost:
            try:
                enhanced_prompt = enhance_prompt_with_ollama(prompt, ollama_model)
                current_prompt = enhanced_prompt
                print(f"Batch {i+1} - Enhanced prompt: {current_prompt}")
            except Exception as e:
                error_messages.append(f"Failed to enhance prompt for batch {i+1}: {str(e)}")
                print(f"Error enhancing prompt: {e}")
        
        # Generate image
        try:
            response = requests.post(
                "https://api.aimlapi.com/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "prompt": current_prompt,
                    "model": model_name,
                },
                timeout=60  # Add timeout to prevent hanging
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Handle response based on API structure
            if "images" in data:
                # Handle API response with 'images' array containing base64 data
                for img_data in data["images"]:
                    if isinstance(img_data, dict) and "base64" in img_data:
                        img_bytes = base64.b64decode(img_data["base64"])
                    elif isinstance(img_data, str):
                        img_bytes = base64.b64decode(img_data)
                    else:
                        continue
                        
                    img = Image.open(io.BytesIO(img_bytes))
                    # Generate a unique filename
                    filename = f"generated_image_{uuid.uuid4()}.png"
                    img.save(filename)
                    results.append((img, current_prompt))
            elif "data" in data and isinstance(data["data"], list):
                # Handle API response with 'data' array containing URL or base64
                for item in data["data"]:
                    if "url" in item:
                        img_response = requests.get(item["url"])
                        img = Image.open(io.BytesIO(img_response.content))
                        results.append((img, current_prompt))
                    elif "b64_json" in item:
                        img_bytes = base64.b64decode(item["b64_json"])
                        img = Image.open(io.BytesIO(img_bytes))
                        results.append((img, current_prompt))
            elif "image" in data:
                # Handle direct image response
                if isinstance(data["image"], dict) and "base64" in data["image"]:
                    img_bytes = base64.b64decode(data["image"]["base64"])
                elif isinstance(data["image"], str):
                    if data["image"].startswith("http"):
                        img_response = requests.get(data["image"])
                        img_bytes = img_response.content
                    else:
                        img_bytes = base64.b64decode(data["image"])
                else:
                    raise ValueError("Unsupported image format in response")
                    
                img = Image.open(io.BytesIO(img_bytes))
                results.append((img, current_prompt))
            else:
                # If the response structure is unknown, log the raw response
                print(f"Unexpected API response structure: {data}")
                error_messages.append(f"Batch {i+1}: Unexpected API response structure")
                
        except Exception as e:
            error_messages.append(f"Batch {i+1}: {str(e)}")
            print(f"Error generating image: {e}")
        
        # Add a small delay between batches
        if i < batch_count - 1:
            time.sleep(1)
    
    error_text = "\n".join(error_messages) if error_messages else None
    return results, error_text

def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        models_response = ollama.list()
        if 'models' in models_response:
            return [model['name'] for model in models_response['models']]
        return ["llama3.2", "llama3", "mistral"]  # Default models if none found
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return ["llama3.2", "llama3", "mistral"]  # Default models on error

# Define UI
with gr.Blocks(title="AI Image Generator") as demo:
    gr.Markdown("# AI Image Generation UI")
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your image generation prompt here",
                lines=3
            )
            
            with gr.Row():
                api_key_input = gr.Textbox(
                    label="API Key",
                    placeholder="Enter your API key",
                    type="password"
                )
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[
                        "stabilityai/stable-diffusion-xl-base-1.0",
                        "flux/dev",
                        "stability-ai/sdxl",
                        "anthropic/claude-3-haiku",
                        "runwayml/stable-diffusion-v1-5"
                    ],
                    value="stabilityai/stable-diffusion-xl-base-1.0"
                )
            
            with gr.Row():
                batch_input = gr.Number(
                    label="Number of Batches",
                    value=1,
                    minimum=1,
                    maximum=10,
                    step=1
                )
                
                with gr.Column():
                    use_prompt_boost = gr.Checkbox(label="Boost Prompt with Ollama", value=False)
                    ollama_model = gr.Dropdown(
                        label="Ollama Model for Prompt Boosting",
                        choices=get_available_ollama_models(),
                        value="llama3.2",
                        interactive=True,
                        visible=True
                    )
            
            # Update Ollama model dropdown visibility based on checkbox
            use_prompt_boost.change(
                fn=lambda x: gr.update(visible=x),
                inputs=use_prompt_boost,
                outputs=ollama_model
            )
            
            generate_btn = gr.Button("Generate Images", variant="primary")
            
        with gr.Column(scale=3):
            output_gallery = gr.Gallery(label="Generated Images").style(grid=2, height="auto")
            output_text = gr.Textbox(label="Status/Errors", interactive=False)
    
    generate_btn.click(
        fn=generate_images,
        inputs=[prompt_input, api_key_input, model_dropdown, batch_input, use_prompt_boost, ollama_model],
        outputs=[output_gallery, output_text]
    )
    
    gr.Markdown("""
    ## How to use
    1. Enter your prompt in the text box
    2. Provide your API key
    3. Select a model from the dropdown
    4. Choose the number of batches to generate
    5. Optionally enable prompt boosting with Ollama
    6. Click "Generate Images" to start the generation process
    
    Note: If using prompt boosting, make sure Ollama is running locally.
    """)

if __name__ == "__main__":
    demo.launch()
