<img width="3188" height="1202" alt="frame (3)" src="https://github.com/user-attachments/assets/517ad8e9-ad22-457d-9538-a9e62d137cd7" />

# VadaScope [VPI- Vada Perfection Index]🎯

## Basic Details

### Team Name: Super Nova

### Team Members

  - Team Lead: Sreehari R - Sree Buddha College Of Engineering, Pattoor
  - Member 2:  Abhinav R - Sree Buddha College Of Engineering, Pattoor

### Project Description

VadaScope is an over-engineered AI that settles the age-old debate of what makes a perfect vada. Using a sophisticated CycleGAN model, our system analyzes an image of a vada and provides an objective, unsolicited Vada Perfection Index (VPI) score, ending family arguments one snack at a time.

### The Problem (that doesn't exist)

Every day, millions of people consume vadas of questionable quality. The lack of a universal, quantifiable metric for vada perfection—from its roundness to its hole-to-vada ratio—threatens the very fabric of our tea-time traditions. This rampant subjectivity is a recipe for culinary chaos and inconsistent snacking experiences.

### The Solution (that nobody asked for)

Our solution is a two-stage hybrid pipeline that combines the predictive power of deep learning with the analytical precision of classical computer vision to deliver the ultimate Vada Perfection Index (VPI) score.

✨Stage 1: The AI Oracle (CycleGAN)
We start by feeding a real-world vada image into a trained CycleGAN model. Acting as an AI Oracle, the model "hallucinates" idealized markings onto the input image — essentially generating the platonic ideal of what the vada should look like.

The result? A marked image representing the Vada Perfection that could have been.

✨Stage 2: The Scrutinizer (CV + VPI Calculation)
The marked image is then passed through a classical Computer Vision module, which extracts four key metrics. These metrics are used to compute the final Vada Perfection Index (VPI\<sub\>S\</sub\>) as follows:

VPI\_S = (w\_size · S\_size + w\_shape · S\_shape + w\_hole · S\_hole + w\_color · S\_color) × 100

📏 Metric Breakdown

| Metric                | Description                                                                                              |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| **S\<sub\>shape\</sub\>** | Measures how closely the vada's outer boundary matches a perfect circle.                                 |
| **S\<sub\>hole\</sub\>**  | Evaluates the position and roundness of the hole, comparing it with the GAN’s ideal.                     |
| **S\<sub\>color\</sub\>** | Measures deviation from the perfect golden-brown color using the **CIEDE2000** color-difference formula. |
| **S\<sub\>size\</sub\>**  | Penalizes vadas that are too big or small, using a Gaussian function centered on the ideal diameter.     |

🎚️Weight Calibration
w\_size  = 0.01  
w\_shape = 0.40  
w\_hole  = 0.30  
w\_color = 0.29

## Technical Details

### Technologies/Components Used

For Software:

  - Languages used: Python
  - Frameworks used: PyTorch, Flask
  - Libraries used:
      - OpenCV (`opencv-python`) for computer vision analysis
      - Pillow (PIL) for image manipulation and report generation
      - Albumentations for data augmentation
      - NumPy for numerical operations
  - Tools used:
      - Git / GitHub for version control
      - Visual Studio Code

For Hardware:

  - Main components: A PC/Laptop with a dedicated NVIDIA GPU. A camera (e.g., phone camera) to capture images of vadas.
  - Specifications: A CUDA-enabled NVIDIA GPU is required for model training (e.g., NVIDIA GeForce RTX 4050 Laptop GPU).
  - Tools required: Not applicable.

### Implementation

For Software:

#### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sree14hari/Useless-Project.git
    ```
2.  **Set up a Virtual Environment:**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install Flask torch torchvision opencv-python Pillow matplotlib
    ```

#### Run

1.  **Start the Backend Server:**
    ```bash
    python app.py
    ```
2.  **Access the Application:**
    Open a web browser and navigate to `http://127.0.0.1:5000`.

### Project Documentation

For Software:

#### Screenshots 

<img width="1917" height="990" alt="image" src="https://github.com/user-attachments/assets/0de5b698-972d-4e34-887c-3a7b2a55ae65" />

<img width="1897" height="985" alt="image" src="https://github.com/user-attachments/assets/49ce93ad-848b-4ada-a9c3-ba80bd3e33c5" />

<img width="1200" height="700" alt="report_2b8a4b29f2bd41bda2b41b1817059ab9" src="https://github.com/user-attachments/assets/68c0be8d-04d3-4ed9-a051-5e1a0dd6a13b" />


#### Diagrams



### Project Demo

#### Video

https://github.com/user-attachments/assets/0d0acf19-0593-4760-ae34-c46289d867f2

#### Additional Demos



https://github.com/user-attachments/assets/d4f79607-52d5-4537-8e83-4a740e043168



## Team Contributions

  - **Sreehari R**: Led the backend development, including setting up the Flask server, training the CycleGAN model, and implementing the computer vision logic for VPI calculation.
  - **Abhinav R**: Designed and developed the main conceptual idea of the project and the frontend user interface, including the HTML structure, all CSS styling and animations, and the JavaScript logic for handling user interaction and displaying results.

-----

Made with ❤️ at TinkerHub Useless Projects
