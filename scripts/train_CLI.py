from ultralytics import YOLO
from ultralytics import settings
import torch
import pandas as pd
import torch
import os
import glob
import wandb
import yaml
import ultralytics
import cv2
print('ultralytics %s' % ultralytics.__version__)
print('wandb %s' % wandb.__version__)
from wandb.integration.ultralytics import add_wandb_callback
from datetime import datetime

def train_tune(project_name, name,config_path,model_path):
    # Initialize wandb
    wandb.init(project=project_name, name=name)
    
    # Load configuration from yaml file
    with open('configs/train_cfg.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create YOLO model
    
    model = YOLO('models/yolov8n.pt') 
     
    # Add wandb callback with model checkpointing enabled
    add_wandb_callback(model,enable_model_checkpointing=True)
    
    # Run hyperparameter tuning with config
    results = model.tune(**cfg, iterations= 300) #number of iterations)
    
    model.save('models/yolov8n_tune.pt')
    wandb.finish()

def main():
    #prompt project name and name
    project_name = input("Enter project name: ")
    name = input("Enter name: ")
    config_path = 'configs/train_cfg.yaml'
    model_path = 'models/yolov8n.pt' 
    # Run hyperparameter tuning 
    train_tune(project_name, name,config_path,model_path)

if __name__ == "__main__":
    main() 