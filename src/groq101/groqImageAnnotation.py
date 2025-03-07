import os
import base64
import gradio as gr
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from groq import Groq
from tqdm import tqdm
import time

def encode_image_to_base64(image_path):
    """Encode an image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_images(folder_path, api_key, batch_size=5, sleep_time=2):
    """Process all images in the folder and create annotations using Groq API"""
    if not api_key:
        raise ValueError("Groq API key is required")
        
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Get all image files from the folder
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f)) and 
        os.path.splitext(f)[1].lower() in image_extensions
    ]
    
    if not image_files:
        return "No image files found in the selected folder", None
    
    results = []
    
    # Process images in batches to avoid rate limiting
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch = image_files[i:i+batch_size]
        
        for img_path in batch:
            try:
                # Convert image to base64
                base64_image = encode_image_to_base64(img_path)
                
                # Call Groq API
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Use appropriate vision model
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Please describe this image in detail and provide the following annotations: 1) Main subjects, 2) Background elements, 3) Colors, 4) Mood/atmosphere, 5) Any text visible, 6) Key objects"},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]
                        }
                    ],
                    temperature=0.5,
                    max_tokens=500
                )
                
                # Extract response
                response = completion.choices[0].message.content
                
                # Get image metadata
                img = Image.open(img_path)
                width, height = img.size
                file_size = os.path.getsize(img_path)
                file_name = os.path.basename(img_path)
                
                # Add to results
                results.append({
                    'file_path': img_path,
                    'file_name': file_name,
                    'width': width,
                    'height': height,
                    'file_size': file_size,
                    'annotation': response
                })
                
            except Exception as e:
                results.append({
                    'file_path': img_path,
                    'file_name': os.path.basename(img_path),
                    'width': None,
                    'height': None,
                    'file_size': None,
                    'annotation': f"Error: {str(e)}"
                })
            
        # Sleep between batches to avoid rate limits
        if i + batch_size < len(image_files):
            time.sleep(sleep_time)
    
    # Create DataFrame and save as Parquet
    df = pd.DataFrame(results)
    
    # Generate output path in the same directory as input folder
    output_path = os.path.join(os.path.dirname(folder_path), 
                              f"{os.path.basename(folder_path)}_annotations.parquet")
    
    # Save as parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    
    return f"Processed {len(results)} images. Results saved to {output_path}", df

# Create Gradio interface
with gr.Blocks(title="Groq Vision Image Annotator") as app:
    gr.Markdown("# Image Annotation with Groq Vision API")
    gr.Markdown("Upload a folder of images to generate annotations using Groq's Vision model and save as a Parquet dataset.")
    
    with gr.Row():
        with gr.Column():
            api_key_input = gr.Textbox(label="Groq API Key", placeholder="Enter your Groq API key", type="password")
            folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the full path to your image folder")
            folder_button = gr.Button("Browse")
            batch_size = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Batch Size")
            sleep_time = gr.Slider(minimum=0, maximum=10, value=2, step=0.5, label="Sleep Time Between Batches (seconds)")
            process_button = gr.Button("Process Images")
        
    output_text = gr.Textbox(label="Status")
    output_table = gr.DataFrame(label="Results")
    
    # File browser to select a folder
    def select_folder():
        folder_path = gr.Files(label="Select a folder", file_count="directory")
        return folder_path
    
    folder_button.click(fn=select_folder, outputs=folder_input)
    
    # Process button
    process_button.click(
        fn=process_images, 
        inputs=[folder_input, api_key_input, batch_size, sleep_time], 
        outputs=[output_text, output_table]
    )

# Launch the app
if __name__ == "__main__":
    app.launch()
