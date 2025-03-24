import cv2
import os
import numpy as np

input_folder = r"newspaper_images\06\01\MAL"  # Folder containing images
output_folder = "annotated_images/06/01/MAL/annotated_data"  # Folder to save text files
output_img_folder = "annotated_images/06/01/MAL/annotated_images"  # Folder to save annotated images

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(output_img_folder, exist_ok=True)

# Get list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]

def non_max_suppression(boxes, overlapThresh=0.3):
    """Applies Non-Maximum Suppression (NMS) to remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas and sort by bottom-right Y-coordinate
    areas = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    selected_idxs = []
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        selected_idxs.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / areas[idxs[:last]]

        # Remove indexes with overlap above threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[selected_idxs].tolist()

for image_file in image_files:
    img_path = os.path.join(input_folder, image_file)
    
    # Read image
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[:2]
    img_copy = img.copy()  # Copy for drawing

    # Convert to grayscale & apply edge detection
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edge_img = cv2.Canny(blur_img, 20, 200)

    # Find contours (use RETR_EXTERNAL to get only outermost contours)
    contours, _ = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare bounding boxes
    detected_boxes = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:  # Keep only rectangular shapes
            min_area = 1500
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                detected_boxes.append((x, y, x + w, y + h))

    # Apply Non-Maximum Suppression (NMS)
    filtered_boxes = non_max_suppression(detected_boxes)

    # Store normalized bounding boxes
    normalized_bounding_boxes = set()
    
    for x1, y1, x2, y2 in filtered_boxes:
        x_center = round(((x1 + x2) / 2) / img_width, 6)
        y_center = round(((y1 + y2) / 2) / img_height, 6)
        width = round((x2 - x1) / img_width, 6)
        height = round((y2 - y1) / img_height, 6)

        normalized_bounding_boxes.add((x_center, y_center, width, height))

        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display normalized coordinates on image
        text = f"{x_center:.2f}, {y_center:.2f}, {width:.2f}, {height:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_width, text_height = text_size

        # Rectangle behind text for better readability
        cv2.rectangle(img_copy, (x1, y1 - 20), (x1 + text_width, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Extract filename without extension
    image_name = os.path.splitext(image_file)[0]

    # Save bounding box coordinates to text file (YOLO format)
    txt_filename = os.path.join(output_folder, f"{image_name}.txt")
    with open(txt_filename, "w") as f:
        for box in normalized_bounding_boxes:
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")

    # Save the annotated image
    output_img_path = os.path.join(output_img_folder, f"{image_name}_annotated.png")
    cv2.imwrite(output_img_path, img_copy)

    print(f"Processed {image_file} - Bounding boxes saved to {txt_filename}")
    print(f"Annotated image saved to {output_img_path}")

print("Processing complete!")