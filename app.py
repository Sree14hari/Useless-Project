from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
import os
from uuid import uuid4
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont # <-- NEW IMPORT

from gan_model import ResnetGenerator
from vada_analysis import rate_my_vada, create_visual_report

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'models/35_net_G.pth'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()
print("✅ VadaScope GAN model loaded successfully.")

def run_gan(input_path, output_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(input_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image_data = ((output_tensor.squeeze().permute(1, 2, 0).numpy() + 1) / 2.0 * 255.0).clip(0, 255)
    result = Image.fromarray(output_image_data.astype('uint8'))
    result.save(output_path)

# --- NEW CERTIFICATE GENERATION ROUTE ---
@app.route('/generate_certificate')
def generate_certificate():
    name = request.args.get('name', 'Valued Vada Connoisseur')
    score = request.args.get('score', '0.00')
    
    template_path = 'certificate_template.png'
    font_path = 'fonts/Poppins-Bold.ttf'
    
    # Create certificate
    image = Image.open(template_path)
    draw = ImageDraw.Draw(image)
    
    # Define fonts
    try:
        name_font = ImageFont.truetype(font_path, 60)
        text_font = ImageFont.truetype(font_path, 40)
        score_font = ImageFont.truetype(font_path, 70)
    except IOError:
        return "Font file not found. Make sure Poppins-Bold.ttf is in the 'fonts' directory.", 500

    # Define text and positions
    # These coordinates (x, y) might need adjustment based on your template
    congrats_text = "is hereby certified as a"
    title_text = "Master Vada Analyst"
    score_text = f"With a VPI Score of {score}"

    # Draw text on the image
    draw.text((600, 250), name, font=name_font, fill="black", anchor="ms")
    draw.text((600, 320), congrats_text, font=text_font, fill="black", anchor="ms")
    draw.text((600, 380), title_text, font=name_font, fill="black", anchor="ms")
    draw.text((600, 500), score_text, font=score_font, fill="#d35400", anchor="ms")
    
    # Save the certificate to a temporary path
    cert_filename = f"certificate_{uuid4().hex}.png"
    cert_path = os.path.join(RESULT_FOLDER, cert_filename)
    image.save(cert_path)
    
    return send_file(cert_path, as_attachment=True)
# ---------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'vada_image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file part.'})
    file = request.files['vada_image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file.'})

    unique_id = uuid4().hex
    filename = f"{unique_id}.jpg"
    gan_output_filename = f"gan_output_{unique_id}.png"
    report_filename = f"report_{unique_id}.png"

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    gan_output_path = os.path.join(RESULT_FOLDER, gan_output_filename)
    file.save(input_path)

    try:
        run_gan(input_path, gan_output_path)
        results = rate_my_vada(gan_output_path, output_dir=RESULT_FOLDER)
        if results is None:
             raise Exception("Could not analyze the vada. Please try a clearer image.")

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