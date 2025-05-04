import os
import io
# Workaround for PIL/Gradio bug :contentReference[oaicite:13]{index=13}
import PIL.Image
import gradio as gr
from gradio_client import Client, handle_file

from gradio_client.client import re
from numpy import array

#variables
title = "V_TRY"

PORT = int(os.environ.get("PORT", 8000))

# 1. Load your HF token from env
HF_TOKEN = os.getenv("HF_TOKEN")  # export HF_TOKEN="hf_..."
# 1) Connect to the Leffa Gradio app’s predict endpoint
# Use the full "/call/predict" API path as shown on the View API page
client = Client(
    "franciszzj/Leffa",
    hf_token=HF_TOKEN,
)  # Gradio Python client


def virtual_tryon(
    person_path,
    garment_path,
    garment_type,
):
    # 2) Wrap file inputs so Gradio client uploads them correctly
    person_file = handle_file(
        person_path
    )  # handle_file uploads the image :contentReference[oaicite:6]{index=6}
    garment_file = handle_file(garment_path)

    # 3) Build inputs in the exact order shown on the “Use via API” page :contentReference[oaicite:7]{index=7}

    # 4) Call the named endpoint with handle_file inputs
    result = client.predict(
        person_file,  # Person Image
        garment_file,  # Garment Image
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type=garment_type,
        vt_repaint=False,
        api_name="/leffa_predict_vt",
    )
    # result[0] is the generated image filepath on the server
    return result[0]  # Gradio will download & display this file

    # 5) Gradio UI


with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
    gr.Markdown("## V_TRY DEMO")

    #Test for image compression
    file_input = gr.HTML(
        '<input type="file" id="fileElem" accept="image/*" style="display:none" />'
    )

    with gr.Row():
        with gr.Column():
            # gr.Markdown("####UPLOAD PERSON IMAGE")
            src = gr.Image(sources="upload",
                           type="filepath",
                           label="Person Image",
                           elem_id="personImg")
        with gr.Column():
            #gr.Markdown("####UPLOAD GARMENT IMAGE")
            ref = gr.Image(sources="upload",
                           type="filepath",
                           label="Garment Image",
                           elem_id="garmentImg")
        with gr.Column():
            # gr.Markdown("####Select the Garment type")
            garment_type = gr.Radio(
                choices=[("Upper", "upper_body"), ("Lower", "lower_body"),
                         ("Dress", "dresses")],
                value="upper_body",
                label="Garment Type",
            )
        with gr.Column():
            gr.Markdown("##Output Image")
            out = gr.Image(
                type="filepath",
                label="Result",
            )
            with gr.Row():
                btn = gr.Button("Generate")

        #test for image compression
        file_input.change(fn=None,
                          inputs=None,
                          outputs=[src, ref],
        js="""
          if (!window.imageCompression) {
            const s=document.createElement('script');
            s.src='https://cdn.jsdelivr.net/npm/browser-image-compression@1.0.0/dist/browser-image-compression.js';
            document.head.appendChild(s);
            await new Promise(res=>s.onload=res);
          }
          const file=document.getElementById('fileElem').files[0];
          const options={maxSizeMB:1, maxWidthOrHeight:1024, useWebWorker:true};
          const compressed=await imageCompression(file, options);
          const reader=new FileReader();
          reader.onload=()=>{ src.setValue(reader.result); ref.setValue(reader.result); };
          reader.readAsDataURL(compressed);
        """)

        btn.click(virtual_tryon, [src, ref, garment_type], out)

demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    share=False,
    show_error=True,
    pwa=True,
)
