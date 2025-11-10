import csv
import pandas as pd

class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr'])
    
    def log_epoch(self, epoch, train_loss, val_loss, lr):
        with open(self.log_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr])
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}".format(epoch, train_loss, val_loss, lr))
