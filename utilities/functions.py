import os
import torch

import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.nn import functional as F


def mkdict(state_dict, model_dict, vocab, validation_losses, epoch, optimizer, scheduler=None):
    ''' makes a simple dictionary of the given inputs '''
    out = {
        'state_dict': state_dict,
        'model_dict': model_dict,
        'vocab': vocab,
        'validation_losses': validation_losses,
        'epoch': epoch,
        'optimizer': optimizer,
        'scheduler': scheduler,
    }
    return out


def mkdir_model(model_name):
    ''' makes the necessary folders for saving the model states '''
    models_dir = './trained_models'
    pth_dir = os.path.join(models_dir, model_name)
    if not os.path.isdir(models_dir):
        os.mkdir(models_dir)
    if not os.path.isdir(pth_dir):
        os.mkdir(pth_dir)


def train_model(model_list: list, train_loader, valid_loader, vocab, device, criterion=None, optimizer=None, scheduler=None, epoch=100, epoch_s=0, lr=1e-3, patience=30):
    ''' trains a model '''

    # extracting model and model name
    model = model_list[0]
    model_name = model_list[1]

    # extracting model information
    model_dict = {
            'lstm_layers': model.LSTM_layers,
            'hidden_dim': model.hidden_dim,
            'embedding_dim': model.embedding_dim,
            'vocab_size': model.input_size,
            'num_classes': model.num_classes,
            'batch_size': model.batch_size
            }

    # transferring model to the given device
    model = model.to(device)

    # initializing values
    stale = 0
    best_valid_loss = 10000
    break_point = 0

    # initializing optimizer, scheduler and the criterion
    if not optimizer:
        optimizer = optim.RAdam(model.parameters(), lr=lr)
    if not scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch)
    if not criterion:
        criterion = BCEWithLogitsLoss()

    # making required folders
    mkdir_model(model_name)

    # creating lists of losses to return
    Loss_list = []
    Valid_Loss_list = []
    learning_rate_list = []

    # iterating epochs
    for i in range(epoch_s, epoch):
        # training
        print(f'TRAINING [{i + 1:03d}/{epoch:03d}]')
        model.train()
        train_losses = []

        for _, batch in enumerate(tqdm(train_loader)):
            inputs, labels = batch
            outputs = model(inputs.to(device))

            loss = criterion(labels.to(device), outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / len(train_losses)
        Loss_list.append(train_loss)
        print(f"\t[ Train | {i + 1:03d}/{epoch:03d} ] Loss = {train_loss:.5f}")

        scheduler.step()
        for param_group in optimizer.param_groups:
            learning_rate_list.append(param_group["lr"])
            print('\tlearning rate %f' % param_group["lr"])

    # validation
        print(f'VALIDATION [{i + 1:03d}/{epoch:03d}]')
        model.eval()
        valid_losses = []
        for batch in tqdm(valid_loader):
            inputs, labels = batch

            with torch.no_grad():
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
            valid_losses.append(loss.item())

        valid_loss = sum(valid_losses) / len(valid_losses)
        Valid_Loss_list.append(valid_loss)

        break_point = i + 1

        # checking model performance and saving improvements
        if valid_loss < best_valid_loss:
            print(f"\t[ Valid | {i + 1:03d}/{epoch:03d} ]", end=' ')
            print(f"Loss = {valid_loss:.5f} -> New best performance!")
            print(
                f"\t\tSaving this model in './trained_models/{model_name}/best_model.pth'")

            net = mkdict(model.state_dict(), model_dict, vocab,  Valid_Loss_list,
                         epoch, optimizer, scheduler)
            torch.save(net, f'trained_models/{model_name}/best_model.pth')

            best_valid_loss = valid_loss
            stale = 0
        else:
            print(f"\t[ Valid | {i + 1:03d}/{epoch:03d} ]", end=' ')
            print(f"Loss = {valid_loss:.5f}")
            stale += 1
            if stale > patience:
                print(
                    f'\n\t\t[No improvement for {patience} consecutive epochs, stopping early.]')
                break

        # saving the latest model
        net = mkdict(model.state_dict(), model_dict, vocab, Valid_Loss_list,
                     epoch, optimizer, scheduler)
        torch.save(net, f'trained_models/{model_name}/last_model.pth')

    result = {
        'break_point': break_point,
        'Valid_Loss_list': Valid_Loss_list,
        'learning_rate_list': learning_rate_list,
        'Loss_list': Loss_list,
    }

    return result


def keep_keys(dictionary, keys):
    ''' removes any unnecessary keys from the given dictionary '''
    d = {}
    for key in keys:
        d[key] = dictionary[key]
    return d


def plot_loss(break_point, Valid_Loss_list, learning_rate_list, Loss_list):
    ''' plots the evaluations of our training '''
    
    plt.figure(dpi=500)

    plt.subplot(311)
    x = range(break_point)
    y = Loss_list
    plt.plot(x, y, 'ro-', label='Train Loss')
    plt.ylabel('Train Loss')
    plt.xlabel('epochs')

    plt.subplot(312)
    plt.plot(range(break_point), Valid_Loss_list, 'bs-', label='Valid Loss')
    plt.ylabel('Validation Loss')
    plt.xlabel('epochs')

    plt.subplot(313)
    plt.plot(x, learning_rate_list, 'ro-', label='Learning rate')
    plt.ylabel('Learning rate')
    plt.xlabel('epochs')

    plt.legend()
    plt.show()


def model_inference(model, input, device, thresh=.5):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(input.to(device))
        output = F.sigmoid(output)
        labels = (output > thresh).type(torch.int)
    return labels  