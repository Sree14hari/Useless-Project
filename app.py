from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import os
from uuid import uuid4

# --- FIX 1: Corrected imports ---
import torch
import torchvision.transforms as transforms
from PIL import Image

from gan_model import ResnetGenerator
from vada_analysis import rate_my_vada, create_visual_report

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'models/35_net_G.pth'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- FIX 2: Correctly instantiate the model with all parameters ---
# This must match the architecture in gan_model.py
model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("✅ VadaScope GAN model loaded successfully.")

# --- FIX 3: Replaced with the correct image processing pipeline ---
def run_gan(input_path, output_path):
    """
    Correctly transforms and runs an image through the GAN model.
    """
    # Define the same transformations used during training
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = Image.open(input_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Un-normalize and save the output
    output_image_data = ((output_tensor.squeeze().permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0).clip(0, 255)
    result = Image.fromarray(output_image_data.astype('uint8'))
    result.save(output_path)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'vada_image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file part.'})
    file = request.files['vada_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file.'})

    # Use a unique filename to prevent browser caching issues
    unique_id = uuid4().hex
    filename = f"{unique_id}.jpg"
    gan_output_filename = f"gan_output_{unique_id}.png"
    report_filename = f"report_{unique_id}.png"

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    gan_output_path = os.path.join(RESULT_FOLDER, gan_output_filename)
    file.save(input_path)

    try:
        # Step 1: Run image through the GAN
        run_gan(input_path, gan_output_path)

        # Step 2: Analyze the GAN's output to get VPI
        results = rate_my_vada(gan_output_path, output_dir=RESULT_FOLDER)
        if results is None:
             raise Exception("Could not analyze the vada. Please try a clearer image.")

        # Step 3: Generate the visual report dashboard
        # Using show=False prevents the server from crashing
        create_visual_report(results, output_dir=RESULT_FOLDER, report_filename=report_filename, show=False)

        return jsonify({
            'success': True,
            'vpi_score': results['VPI_S'],
            'annotated_image_url': f"/static/results/{os.path.basename(results['annotated_image_path'])}",
            'report_image_url': f"/static/results/{report_filename}"
        })
    except Exception as e:
        app.logger.error(f"Analysis failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)