import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import face_recognition
import matplotlib.pyplot as plt
import io
import base64
import torch.nn as nn  # Add this line
from .model import Model


im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax(dim=1)
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))

def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    return image

def predict(model, img):
  fmap, logits = model(img)
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item() * 100
  return int(prediction.item()), confidence

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1  # We process one video at a time

    def __getitem__(self, idx):
        frames = []
        frame_count = 0
        for frame in self._frame_extract(self.video_path):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except IndexError:
                pass  # Handle cases where no face is detected
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
            frame_count += 1
            if len(frames) == self.count:
                break
        if not frames:  # Handle cases where no frames were extracted
            raise ValueError("Could not extract enough frames from the video.")
        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def _frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def get_prediction(video_path, model_path):
    device = torch.device("cpu")
    model = Model(2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    video_dataset = ValidationDataset(video_path, sequence_length=20, transform=train_transforms)
    data_loader = torch.utils.data.DataLoader(video_dataset, batch_size=1, shuffle=False)
    try:
        for video_batch in data_loader:
            with torch.no_grad():
                prediction, confidence = predict(model, video_batch.squeeze(0))
                return prediction, confidence
    except ValueError as e:
        return -1, 0.0 # Indicate an error in frame extraction
