from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from io import BytesIO
from flask_cors import CORS

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 2)  # 2 classes: cat and dog

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the model
model = CNN()
model.load_state_dict(torch.load('catdog_classifier.pth'))
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define a route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if the request contains an uploaded file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an image
    if file and allowed_file(file.filename):
        # Read the image file
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes))
        
        # Preprocess the image
        image = transform(image).unsqueeze(0)  # add batch dimension

        # Make predictions
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

        # Return the prediction result
        if predicted.item() == 0:
            return jsonify({'prediction': 'cat'})
        else:
            return jsonify({'prediction': 'dog'})

    return jsonify({'error': 'Invalid file format'})

# Run the Flask app
if __name__ == '__main__':
    app.run()