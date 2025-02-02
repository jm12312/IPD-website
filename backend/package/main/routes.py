from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import numpy as np
import librosa
import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
main_bp = Blueprint('main', __name__)

@main_bp.route("/")
def home():
    return jsonify({"msg": "Hi"})

def extract_features(file_path=None, max_length=270, trans=None, segment=None):
    """
    Extracts MFCC features from an audio file and pads/truncates them to a fixed length.
    
    Parameters:
    - file_path (str): Path to the audio file.
    - max_length (int): The length to which the MFCC matrix should be padded or truncated (number of time frames).
    
    Returns:
    - np.ndarray: A flattened array of padded/truncated MFCC features.
    """
    # Load the audio file with librosa
    if file_path:
      y, sr = librosa.load(file_path, sr=22050)
    else:
      y = segment
      sr = 22050

    if trans != None:
        y = trans(y)
    
    # Extract MFCC features (13 MFCCs)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # Pad or truncate MFCCs to the specified max_length (number of time frames)
    if mfcc.shape[1] < max_length:
        # Pad with zeros if MFCC matrix is smaller than max_length
        pad_width = max_length - mfcc.shape[1]
        mfcc_padded = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if MFCC matrix is larger than max_length
        mfcc_padded = mfcc[:, :max_length]

    # Compute the mean, min, and max across each feature (MFCC coefficient) dimension
    mean_features = np.mean(mfcc_padded, axis=1)  # Mean for each MFCC coefficient
    min_features = np.min(mfcc_padded, axis=1)    # Min for each MFCC coefficient
    max_features = np.max(mfcc_padded, axis=1)    # Max for each MFCC coefficient

    rms = librosa.feature.rms(y=y)
    
    # Ensure the padding width is not negative
    pad_width_rms = max(0, max_length - rms.shape[1])
    rms_padded = np.pad(rms, ((0, 0), (0, pad_width_rms)), mode='constant')[:, :max_length]
    
    # Compute ZCR (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    
    # Ensure the padding width is not negative
    pad_width_zcr = max(0, max_length - zcr.shape[1])
    zcr_padded = np.pad(zcr, ((0, 0), (0, pad_width_zcr)), mode='constant')[:, :max_length]
   
    
    # Flatten the padded MFCC features into a single vector for each file
    flattened_mfcc = mfcc_padded.flatten()
    flattened_rms = rms_padded.flatten()
    flattened_zcr = zcr_padded.flatten()

    # Combine the flattened MFCC features with the mean, min, and max statistics
    return np.concatenate([flattened_mfcc, mean_features, min_features, max_features, flattened_rms, flattened_zcr])

def accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class Emotion_FCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the convolution layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)  # First convolution (5x kernel)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)  # Second convolution (5x kernel)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)  # Third convolution (5x kernel)
        
        # MaxPooling to reduce dimensionality after each Conv layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        # Adaptive pooling to reduce the dimensionality to a single value for each feature map
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Output size will be (batch_size, channels, 1)

        # Fully connected layer to map the features to the output (7 classes)
        self.fc = nn.Linear(256, 7)  # Output layer for 7 classes

    def forward(self, x):
        # Ensure input is shaped (batch_size, 1, 6000)
        x = x.view(x.size(0), 1, -1)  # Reshaping to (batch_size, 1, num_features)
        
        # Apply convolutional layers with ReLU activations and max pooling
        x = F.relu(self.conv1(x))  # First Conv layer
        x = self.pool(x)  # Max Pooling

        x = F.relu(self.conv2(x))  # Second Conv layer
        x = self.pool(x)  # Max Pooling

        x = F.relu(self.conv3(x))  # Third Conv layer
        x = self.pool(x)  # Max Pooling

        # Apply Global Average Pooling to reduce each feature map to a single value
        x = self.global_pool(x)  # Shape will be (batch_size, num_channels, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_channels)

        # Fully connected layer (classification)
        x = self.fc(x)  # Output logits for 7 classes
        
        return x
    # def __init__(self):
    #     super().__init__()
    #     # Using BatchNorm and Dropout for better generalization
    #     self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)  # First convolution
    #     self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # Second convolution
    #     self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)  # Third convolution

    #     # MaxPooling to reduce dimensionality
    #     self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

    #     # Global Average Pooling
    #     self.global_pool = nn.AdaptiveAvgPool1d(1)  # This will reduce the output of Conv layers to a single value per feature map

    #     # Fully connected layer to map the features to the output (7 classes)
    #     self.fc = nn.Linear(256, 7)  # Only one linear layer for classification

    # def forward(self, x):
    #     x = x.view(x.size(0), 1, -1)  # Reshaping to (batch_size, 1, num_features) for Conv1D

    #     # Apply 1D convolutions with ReLU activations
    #     x = F.relu(self.conv1(x))  # First Conv layer
    #     x = self.pool(x)  # Max Pooling

    #     x = F.relu(self.conv2(x))  # Second Conv layer
    #     x = self.pool(x)  # Max Pooling

    #     x = F.relu(self.conv3(x))  # Third Conv layer
    #     x = self.pool(x)  # Max Pooling

    #     # Apply Global Average Pooling to reduce each feature map to a single value
    #     x = self.global_pool(x)  # Output shape will be (batch_size, num_channels, 1)
    #     x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_channels)

    #     # Final classification layer
    #     x = self.fc(x)  # Output logits for 7 classes
    #     return x

    def calculate_loss(self, batch):
        x, y = batch
        op = self(x)
        if torch.isnan(op).any():
          print("NaN detected in model output")
          print(op)
        # y = (y-1).long()
        y = y.long()
        # print(y)
        loss = F.cross_entropy(op, y)
        return loss

    def validation_step(self, batch):
        x, y = batch
        op = self(x)
        y = y.long()
        loss = F.cross_entropy(op, y)
        acc = accuracy(op, y)

        # Add logging to see what is returned
        # print(f"Validation step - loss: {loss.item()}, acc: {acc.item()}")

        return {'val_loss': loss, 'val_acc': acc}


    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f'Epoch: {epoch},  Val Loss: {result["val_loss"]}, Val Acc: {result["val_acc"]}')

dic = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}
model_audio = Emotion_FCNN()
model_audio.load_state_dict(torch.load('package/assets/models/emotion_classification_audio_CNN_np.pth', map_location=torch.device('cpu')))

@main_bp.route("/audio/emotions", methods=["POST"])
def audio_emotions():
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_file"]
    model_audio.eval()
    audio_path = "package/assets/audio/tem-audio.wav"
    audio_file.save(audio_path)
    y, sr = librosa.load(audio_path, sr=22050)

    # Define the window length (3 seconds)
    window_size = 5 * sr  # 3 seconds in samples
    hop_length = 4 * sr # non-overlapping

    # Split the audio into chunks
    segments = []
    for start in range(0, len(y), hop_length):
        end = start + window_size
        segment = y[start:end]
        # print(segment.shape)
        if len(segment) < window_size:
            padding = np.zeros(window_size - len(segment))
            segment = np.concatenate((segment, padding))
        if len(segment) == window_size:  # Make sure it's a full segment
            segments.append(segment)

    segments = np.array(segments)
    # List to store the results
    results = []

    for i in range(segments.shape[0]):
        inp_features = extract_features(segment=segments[i])
        inp_features = torch.tensor(inp_features, dtype=torch.float32)
        inp_features = inp_features.view(1, 1, -1)
        prediction = torch.max(model_audio(inp_features), dim=1).indices.item()
        
        overlap_start = i * 4  # Start of the current segment
        overlap_end = overlap_start + 5
        
        # Append the result as a tuple
        results.append((dic[prediction], overlap_start, overlap_end))

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results, columns=["Prediction", "Start_time", "End_time"])

    return jsonify(results_df.to_dict(orient="records"))


