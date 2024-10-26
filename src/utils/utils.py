from torchvision.datasets import MNIST
import os
from pathlib import Path
import csv
import sys
import argparse

def save_dataset(dataset, data_location, set_name):
    # Create directory structure: data_location/MNIST/set_name (train or test)
    foldername = data_location / "MNIST" / set_name
    foldername.mkdir(parents=True, exist_ok=True)
    
    # Create CSV file path
    csv_path = data_location / "MNIST" / f"{set_name}.csv"
    
    # Open CSV file and write data
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'label', 'named_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Iterate through dataset
        for idx, (data, label) in enumerate(dataset):
            named_label = dataset.classes[label]
            image_path = foldername / f"{idx}.png"
            data.save(image_path)
            writer.writerow({'filename': f"{idx}.png", 'label': label, 'named_label': named_label})

def load_and_create_datasets(data_location, logging_off):
    # If logging is off, redirect output to null
    if logging_off:
        f = open(os.devnull, "w")
        sys.stdout = f
        
    # Convert data_location to Path object
    data_location = Path(data_location)
    print(f"Storing data at {data_location.resolve()}.")

    # Download training and test datasets
    train_data = MNIST(root=data_location, train=True, download=True)
    test_data = MNIST(root=data_location, train=False, download=True)

    # Save both datasets
    save_dataset(train_data, data_location, "train")
    save_dataset(test_data, data_location, "test")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser("Download MNIST from PyTorch and move the raw data into a folder.")
    parser.add_argument("dir", type=str, help="Folder to store images.")
    parser.add_argument("--logging_off", action="store_true", help="Turn off logging.")
    parser.set_defaults(logging_off=False)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run main function
    load_and_create_datasets(args.dir, args.logging_off)
