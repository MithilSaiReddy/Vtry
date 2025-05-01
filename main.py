import os
import io
import PIL.Image       # Workaround for PIL/Gradio bug :contentReference[oaicite:13]{index=13}
import gradio as gr
from gradio_client import Client, handle_file

from numpy import array
# 1. Load your HF token from env
HF_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN="hf_..."
# 1) Connect to the Leffa Gradio app’s predict endpoint
# Use the full "/call/predict" API path as shown on the View API page
client = Client("franciszzj/Leffa", hf_token=HF_TOKEN, )  # Gradio Python client

def virtual_tryon(person_path, garment_path):
        # 2) Wrap file inputs so Gradio client uploads them correctly
    person_file = handle_file(person_path)   # handle_file uploads the image :contentReference[oaicite:6]{index=6}
    garment_file = handle_file(garment_path)

        # 3) Build inputs in the exact order shown on the “Use via API” page :contentReference[oaicite:7]{index=7}

        # 4) Call the named endpoint with handle_file inputs
    result = client.predict(
        person_file,      # Person Image
        garment_file,     # Garment Image
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        api_name="/leffa_predict_vt"     
    )
        # result[0] is the generated image filepath on the server
    return result[0]  # Gradio will download & display this file

    # 5) Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Leffa Virtual Try-On")
    with gr.Row():
        src = gr.Image(sources="upload", type="filepath", label="Person Image")
        ref = gr.Image(sources="upload", type="filepath", label="Garment Image")
        out = gr.Image(type="filepath", label="Result", )
    btn = gr.Button("Generate")
    btn.click(virtual_tryon, [src, ref], out)

demo.launch(share=True,
            show_error=True,
            pwa=True,)
