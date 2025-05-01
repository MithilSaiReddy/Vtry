import gradio as gr  # Gradio for UI and hosting the app
import numpy as np
from gradio_client import Client, file  # Gradio client to call remote Space API

# 1) Instantiate the client for the Leffa Space on Hugging Face
#    - Uses your HF_TOKEN environment variable if set
client = Client("franciszzj/Leffa", hf_token="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")

# 2) Define the virtual-try-on function that wraps the Space's API endpoint
#    - Accepts a person image and a garment image, returns the generated try-on
#    - Uses the `/leffa_predict_vt` endpoint as discovered via client.view_api()
def virtual_tryon(person_img, garment_img):
    # Wrap image inputs for upload
    src = file(person_img)
    ref = file(garment_img)
    # Call the Leffa VT endpoint
    output = client.predict(
        src_image=src,
        ref_image=ref,
        ref_acceleration=False,
        step=30,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False,
        preprocess_garment=False,
        api_name="/leffa_predict_vt",
    )
    # output is (generated_image, mask, densepose)
    return output[0]

# 3) Build the Gradio interface around the function
demo = gr.Interface(
    fn=virtual_tryon,
    inputs=[
        gr.Image(source="upload", label="Person Image"),
        gr.Image(source="upload", label="Garment Image"),
    ],
    outputs=gr.Image(label="Try-On Result"),
    title="Leffa Virtual Try‑On",
    description="Upload a photo of a person and a garment image to see the virtual try-on powered by the Leffa model.",
)

# 4) Launch the app when run as a script
if __name__ == "__main__":
    demo.launch()  # Opens at http://localhost:7860 by default
