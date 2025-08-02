import cv2
import numpy as np
import math
import os
import matplotlib
# --- FIX: Use a non-GUI backend for Matplotlib ---
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# The rate_my_vada function remains the same.
def rate_my_vada(image_path, output_dir='analyzed_results'):
    """
    Analyzes a vada and calculates the Standardized VPI (VPI-S) on a 0-100 scale.
    """
    # --- 1. Load Image ---
    if not os.path.exists(image_path):
        print(f"ERROR: File not found at '{image_path}'")
        return None
    color_image = cv2.imread(image_path)
    if color_image is None:
        return None
    
    annotated_image = color_image.copy()
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    img_height, img_width = color_image.shape[:2]

    # --- 2. GrabCut for the OUTER CONTOUR ---
    mask = np.zeros(color_image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (int(img_width*0.1), int(img_height*0.1), int(img_width*0.8), int(img_height*0.8))
    cv2.grabCut(color_image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    binary_mask_grabcut = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    contours, _ = cv2.findContours(binary_mask_grabcut, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    vada_contour = max(contours, key=cv2.contourArea)

    # --- 3. Hybrid Method to Find Hole ---
    vada_mask = np.zeros(gray_image.shape, np.uint8)
    cv2.drawContours(vada_mask, [vada_contour], -1, 255, -1)
    vada_only_gray = cv2.bitwise_and(gray_image, gray_image, mask=vada_mask)
    _, hole_thresh = cv2.threshold(vada_only_gray, 50, 255, cv2.THRESH_BINARY)
    hole_mask = cv2.bitwise_not(hole_thresh)
    hole_mask = cv2.bitwise_and(hole_mask, hole_mask, mask=vada_mask)
    hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    D_hole = 0
    hole_contour = None
    if hole_contours:
        hole_contour = max(hole_contours, key=cv2.contourArea)
        _, radius_hole = cv2.minEnclosingCircle(hole_contour)
        D_hole = 2 * radius_hole
    
    # --- 4. Calculate the Standardized Scores (0-1) ---
    _, radius_vada = cv2.minEnclosingCircle(vada_contour)
    D_avg = 2 * radius_vada
    S_size = D_avg / img_width
    net_area = cv2.countNonZero(vada_mask) - cv2.countNonZero(hole_mask)
    perimeter = cv2.arcLength(vada_contour, True)
    S_shape = (4 * math.pi * net_area) / (perimeter**2) if perimeter > 0 else 0
    S_hole = (1 - (D_hole / D_avg)) if D_avg > 0 else 0
    IDEAL_GOLDEN_BROWN_RATIO = 0.3
    img_hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lower_brown = np.array([10, 80, 50])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(img_hsv, lower_brown, upper_brown)
    brown_pixels = cv2.countNonZero(cv2.bitwise_and(brown_mask, vada_mask))
    rho_gb = brown_pixels / net_area if net_area > 0 else 0
    S_color = 1 - abs(IDEAL_GOLDEN_BROWN_RATIO - rho_gb)

    # --- 5. Calculate Final VPI-S ---
    w_size, w_shape, w_hole, w_color = 0.01, 0.4, 0.3, 0.29
    vpi_s = (w_size*S_size + w_shape*S_shape + w_hole*S_hole + w_color*S_color) * 100
    
    # --- 6. Create Annotated Image ---
    cv2.drawContours(annotated_image, [vada_contour], -1, (0, 255, 0), 3)
    if hole_contour is not None:
        cv2.drawContours(annotated_image, [hole_contour], -1, (0, 0, 255), 3)
    report_text = f"VPI-S: {vpi_s:.2f} / 100"
    cv2.putText(annotated_image, report_text, (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 2)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"annotated_{base_filename}.png")
    cv2.imwrite(save_path, annotated_image)

    return {
        'filename': base_filename, 'VPI_S': vpi_s, 'S_size': S_size,
        'S_shape': S_shape, 'S_hole': S_hole, 'S_color': S_color,
        'annotated_image_path': save_path
    }

# --- FIX: Added 'show' and 'report_filename' parameters ---
def create_visual_report(results, output_dir='analyzed_results', report_filename='report.png', show=True):
    """Creates a dashboard-style visual report of the VPI analysis."""
    vpi_score = results['VPI_S']
    scores = {'Color': results['S_color'], 'Hole': results['S_hole'], 'Shape': results['S_shape'], 'Size': results['S_size']}
    labels = list(scores.keys())
    values = list(scores.values())
    
    fig = plt.figure(figsize=(12, 7), facecolor='#F0F0F0')
    fig.suptitle('Vada Perfection Index - Official Report', fontsize=20, weight='bold')
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    
    ax_img = fig.add_subplot(gs[:, 0])
    annotated_img = cv2.imread(results['annotated_image_path'])
    ax_img.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    ax_img.axis('off')
    ax_img.set_title('Visual Analysis', fontsize=14, style='italic')

    ax_gauge = fig.add_subplot(gs[0, 1])
    ax_gauge.set_facecolor('#F0F0F0')
    ax_gauge.set_title('Final VPI-S Score', fontsize=14, style='italic')
    color = 'red' if vpi_score < 40 else ('orange' if vpi_score < 75 else 'green')
    ax_gauge.add_patch(patches.Wedge((0.5, 0.4), 0.4, 0, 180, facecolor='#DDDDDD', width=0.1))
    ax_gauge.add_patch(patches.Wedge((0.5, 0.4), 0.4, 0, vpi_score * 1.8, facecolor=color, width=0.1))
    ax_gauge.text(0.5, 0.4, f'{vpi_score:.2f}', ha='center', va='center', fontsize=30, weight='bold')
    ax_gauge.text(0.5, 0.25, '/ 100', ha='center', va='center', fontsize=12)
    ax_gauge.set_xlim(0, 1)
    ax_gauge.set_ylim(0, 1)
    ax_gauge.axis('off')

    ax_bar = fig.add_subplot(gs[1, 1])
    ax_bar.set_facecolor('#F0F0F0')
    ax_bar.barh(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title('Component Scores', fontsize=14, style='italic')
    ax_bar.tick_params(left=False, labelbottom=False)
    for index, value in enumerate(values):
        ax_bar.text(value + 0.02, index, f'{value:.2f}', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    report_path = os.path.join(output_dir, report_filename)
    fig.savefig(report_path)
    print(f"Visual report saved to: {report_path}")
    
    # --- FIX: Only show plot if not running on server ---
    if show:
        plt.show()
    plt.close(fig) # Close the figure to free up memory