import gradio as gr
import requests
import cv2

# URL of the API created with FastAPI
API_URL = "https://lab3-nuj8.onrender.com"

# Function to execute when clicking the "Predict button"
def predict(image):
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        _, img_encoded = cv2.imencode(".jpg", image_bgr)
        files = {"file": ("image.jpg", img_encoded.tobytes(), "image/jpeg")}

        response = requests.post(f"{API_URL}/predict", files=files, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("predicted_class")
    except Exception as e:
        return f"Error: {str(e)}"


# GUI creted using Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Image", type="numpy", height=400),
    outputs=gr.Textbox(label="Predicted class"),
    title="Cat/Dog predictor GUI",
    description="Cat/Dog predictor GUI powered by Fastapi + Render + Docker",
)

# Launch the GUI
if __name__ == "__main__":
    iface.launch()