#model import 
from model import DeepCNN as TheModel
#training import
from train import train_model as the_trainer
#predict function 
from predict import classify_images as the_predictor
##dataset class and dataloader content
from dataset import CustomDataset as TheDataset #for dataset having class
from dataset import get_loaders as the_dataloader 
#config parameters
from config import resize_x, resize_y, batch_size, num_epochs, learning_rate
