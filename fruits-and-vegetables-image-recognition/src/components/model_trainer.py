import os
import sys
from dataclasses import dataclass
from datetime import datetime

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts', 'checkpoint.pkl')
    preprocessor_file_path=os.path.join('artifacts', 'model.pkl')
    model_result_file_path=os.path.join('artifacts', f"model_result_{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.png" )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        object = load_object(self.model_trainer_config.preprocessor_file_path)
        self.model = object['model']
        self.criterion = object['criterion']
        self.optimizer = object['optimizer']
        self.device = object['device']
        
    def train(self, device, model, epoch, train_loader):
        model.train()
        
        total_loss = 0.0
        train_accuracy = 0
        
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch} Training")
    
        for batch_idx, (data, targets) in loop:
            data = data.to(device=device)
            targets = torch.tensor(targets).to(device=device)  # Convert targets to a tensor
            
            scores = model(data)
            loss = self.criterion(scores, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            _, preds = torch.max(scores, 1)
            correct_predictions  = torch.sum(preds == targets).item()
            accuracy = correct_predictions / data.size(0)
        
            total_loss += loss.item()
            train_accuracy += accuracy
            
            loop.set_postfix(loss=loss.item(), accuracy=100 * accuracy)
                        
        total_loss /= len(train_loader)
        train_accuracy = 100 * train_accuracy / len(train_loader)
        
        return total_loss, train_accuracy
    
    def test(self, device, model, epoch, test_loader):
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            total_loss = 0
            
            loop = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} Testing")
                        
            for batch_idx, (data, targets) in loop:
                data = data.to(device=device)
                targets = targets.to(device=device)
                output = model(data)
                
                loss = self.criterion(output, targets)
                total_loss += loss.item()
                
                _, preds = torch.max(output, 1)
                total_samples += targets.size(0)
                correct_predictions += (preds == targets).sum().item()
        
                loop.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=100 * correct_predictions / total_samples)
                        
            avg_loss = total_loss / len(test_loader)
            test_accuracy = 100 * correct_predictions / total_samples
            
        return avg_loss, test_accuracy
    
    def plot_result(self, train_accuracies, train_losses, test_accuracies, test_losses):
        # Plot and save the training and testing results
        plt.figure(figsize=(12, 6))
        
        # Training plot
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        
        plt.tight_layout()
        
        plt.savefig(self.model_trainer_config.model_result_file_path)
        
    def initiate_model_trainer(self, train_loader, test_loader):
        try:
            train_losses = []
            train_accuracies = []
            test_losses = []
            test_accuracies = []
            
            logging.info("model load: %s", self.model)
            num_epoch = 50
            best_accuracy = 0
            best_epoch = 0
            best_loss = 0
            for epoch in range(0, num_epoch):
                train_loss  = 0.0
                train_accuracy  = 0
                test_loss  = 0.0
                test_accuracy  = 0
                
                train_loss, train_accuracy = self.train(self.device, self.model, epoch, train_loader)
                test_loss, test_accuracy = self.test(self.device, self.model, epoch, test_loader)

                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                
                logging.info(f"Epoch {epoch+1}/{num_epoch}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
                
                # Save the best model based on validation accuracy
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_epoch = epoch
                    best_loss = test_loss
                    checkpoint_path = self.model_trainer_config.trained_model_file_path
                    
                    # save model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': test_loss,
                        'accuracy': test_accuracy,
                    }, checkpoint_path)
            
            self.plot_result(train_accuracies, train_losses, test_accuracies, test_losses)
            
            return best_epoch, best_accuracy, best_loss
        
        except Exception as e:
            raise CustomException(e, sys)