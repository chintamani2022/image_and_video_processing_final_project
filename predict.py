import torch
from PIL import Image
from torchvision import transforms
from config import resize_x, resize_y

def inferloader(image_paths):
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    images = [transform(Image.open(path).convert('RGB')) for path in image_paths]
    return torch.stack(images)

def classify_images(list_of_img_paths, model):
    model.eval()
    images_batch = inferloader(list_of_img_paths).to(model.device)
    with torch.no_grad():
        outputs = model(images_batch)
        predictions = outputs.argmax(dim=1).cpu().tolist()
        labels = ["Bike" if label == 0 else "Car" for label in predictions]
    return labels
