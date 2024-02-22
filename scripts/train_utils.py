import os
import torch
import tqdm
from constants import WEIGHTS_PATH

def train_model_with_early_stopping(model, train_loader, optimizer, loss_fn, device, epochs, checkpoint_path, min_delta=0.001, patience=5):
    model.train()
    best_loss = float('inf')
    prev_loss = float('inf')
    early_stopping_counter = 0
    for epoch in range(epochs):
        train_loss = 0.0
        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            for i, data in enumerate(tepoch):
                x1, x2, labels = data[0], data[1], data[2]
                x1, x2 = x1.to(device), x2.to(device)
                
                labels = labels.unsqueeze(1)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Convert inputs to float
                x1 = x1.float()
                x2 = x2.float()
                outputs = model(x1, x2)
 
                loss = loss_fn(outputs, labels)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()
                tepoch.set_postfix(loss=train_loss / (i+1))  

        # Check for early stopping conditions
        if prev_loss - train_loss < min_delta:
            early_stopping_counter += 1
        else:
            early_stopping_counter = 0
        prev_loss = train_loss

        # Save model checkpoint
        if checkpoint_path is not None:
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'checkpoint_epoch_{epoch}.pt'))

        # Update best loss and save model if current loss is better
        if train_loss < best_loss:
            best_loss = train_loss
            if checkpoint_path is not None:
                torch.save(model.state_dict(), os.path.join(WEIGHTS_PATH, 'SiameseModel.pt'))

        # Check for early stopping
        if early_stopping_counter >= patience:
            print(f'Early stopping at epoch {epoch} with best loss: {best_loss}')
            break
    return best_loss

def test_model(model, test_loader, loss_fn, device):
    model.eval()
    total_loss_over_epochs = 0.0
    with torch.no_grad():
        test_loss = 0.0
        with tqdm.tqdm(test_loader, unit="batch") as tepoch:
            for i, tdata in enumerate(tepoch):
                x1, x2, tlabels = tdata[0], tdata[1], tdata[2]
                x1, x2 = x1.to(device), x2.to(device)
                tlabels = tlabels.to(device)
                tlabels = tlabels.unsqueeze(1)

                x1 = x1.float()
                x2 = x2.float()
                voutputs = model(x1,x2)
                vloss = loss_fn(voutputs, tlabels)
                
                test_loss += vloss.item()
                tepoch.set_postfix(loss=test_loss / (i+1))   

        total_loss_over_epochs += test_loss
    total_loss_over_epochs /= len(test_loader)
    return total_loss_over_epochs