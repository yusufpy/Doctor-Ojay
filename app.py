from flask import Flask, render_template, request
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw

app = Flask(__name__)

# Load the pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Function to perform object detection, count objects, and draw bounding boxes
def detect_objects(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        #transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)

    boxes = output[0]['boxes']
    scores = output[0]['scores']
    labels = output[0]['labels']

    # Count the number of objects with confidence > 0.5
    object_count = sum(score > 0.5 for score in scores)

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.5:  # Adjust the confidence threshold as needed
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
            draw.text((box[0], box[1]), f"{COCO_INSTANCE_CATEGORY_NAMES[label]}: {score:.2f}", fill="red")

    return image, object_count

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    image = Image.open(file)
    processed_image, object_count = detect_objects(image)

    # Save the processed image with bounding boxes
    processed_image_path = 'static/processed_image.jpg'
    processed_image.save(processed_image_path)

    return render_template('result.html', image_path=processed_image_path, object_count=object_count)

if __name__ == '__main__':
    app.run(debug=True)
