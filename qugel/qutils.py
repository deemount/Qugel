import datetime
import copy
import datetime
import string
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from qblocks.qnns import *

# from qblocks.qutils import *


# !/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project :Quantum-Neural-Network 
@File    :qutils.py
@Author  :SK
@Date    :01.05.2023 16:02 
@Desc    :Utils

'''


def seed_everything(seed):
    # os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_date_postfix():
    """Get a date based postfix for directory name"""
    dt = datetime.datetime.now()
    post_fix = '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    return post_fix


def random_string(string_len=3):
    """Get a random string"""
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(string_len))


def str2bool(v):
    return v.lower() in ['true']


def train_qnn_model(config, criterion, optimizer, qnn):
    for epoch in tqdm(range(config.num_epochs)):
        # print("Epoch {}/{}, Qubits:{}".format(epoch, num_epochs, n_qubits))
        cpu_percent = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().total / (1024 ** 3)
        used_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
        print(
            f"Epoch:[{epoch + 1}/{config.num_epochs}], "
            f"Dataset:{config.data_dir, config.dataset_sizes}, Qubits:{config.n_qubits}, "
            f"RGB:{config.n_RGB},IMG:{config.n_img_w} Layers:{config.n_layers},QNN Params:{[sum(p.numel() for p in qnn.parameters())]} CPU:{cpu_percent}, RAM(GB):{used_ram_gb}/{mem_usage}")

        qnn.train()
        running_loss = 0
        running_corrects = 0
        total_samples = 0

        for batch_idx, (data, target) in tqdm(enumerate(config.dataloaders['train'])):
            data = data.to(config.device)
            target = target.view(-1).to(config.device)
            batch_size = data.size(0)  # Get the actual batch size

            optimizer.zero_grad()
            output = qnn(data)

            # Adjust the output tensor size if necessary
            if output.size(0) > batch_size:
                output = output[:batch_size]

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            running_corrects += torch.sum(predicted == target.data)
            total_samples += batch_size

        batch_loss = running_loss / len(config.dataloaders['train'])
        batch_acc = running_corrects / total_samples
        print(f'[{epoch + 1}] Training Loss: {batch_loss:.3f}, Training Accuracy: {batch_acc:.3f}')

        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_data, val_target in config.dataloaders['val']:
                val_data = val_data.to(config.device)
                val_target = val_target.to(config.device)
                batch_size = val_data.size(0)  # Get the actual batch size

                val_output = qnn(val_data)
                _, val_predicted = torch.max(val_output.data, 1)

                # Adjust the output and target tensors if necessary
                if val_predicted.size(0) > batch_size:
                    val_predicted = val_predicted.narrow(0, 0, batch_size)
                    val_target = val_target.narrow(0, 0, batch_size)

                val_total += batch_size
                val_correct += (val_predicted == val_target).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"[{epoch + 1}] Validation Accuracy: {val_accuracy:.2f}%")


def train_model(config, criterion, optimizer, scheduler, model):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 10000.0  # Large arbitrary number
    best_acc_train = 0.0
    best_loss_train = 10000.0  # Large arbitrary number
    print('Training started:Q={},D={}'.format(config.n_qubits, config.n_layers))
    # print("Corpus:", data_dir, dataset_sizes,class_names)

    torch_dir = 'torch/'

    # try:
    #     Path(torch_dir + mdl_name).mkdir(parents=True, exist_ok=True)
    #     # os.mkdir(torch_dir)
    #     # os.mkdir(torch_dir +mdl_name)
    # except:
    #     print('Ignore directory {} creation failure.'.format(mdl_name))
    #
    # print('MDL:{}, Q-block:{}, Q-bits:{}, Q-depth:{}'.format(mdl_name, model.fc.q_net_block_name, model.fc.n_qubits,
    #                                                          model.fc.q_depth))
    # if plot_circ == True:
    #     Q_Plot(model.fc.q_pqc, model.fc.n_qubits, model.fc.q_depth)
    #     # print(qml.draw(model.fc.q_net)(1.2345,1.2345))
    for epoch in tqdm(range(config.num_epochs)):

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                # Set model to training mode
                model.train()
            else:
                # Set model to evaluate mode
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = config.dataset_sizes[phase] // config.batch_size
            it = 0
            for inputs, labels in config.dataloaders[phase]:
                since_batch = time.time()
                batch_size_ = len(inputs)
                inputs = inputs.to(config.device)
                labels = labels.to(config.device)
                optimizer.zero_grad()

                # Track/compute gradient and make an optimization step only when training
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Print iteration results
                running_loss += loss.item() * batch_size_
                batch_corrects = torch.sum(preds == labels.data).item()
                running_corrects += batch_corrects
                it += 1

            # Print epoch results
            epoch_loss = running_loss / config.dataset_sizes[phase]
            epoch_acc = running_corrects / config.dataset_sizes[phase]
            print('Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}, Batch time: {:.4f} '.format('train' if phase == 'train' else 'val  ',epoch + 1, config.num_epochs, epoch_loss,epoch_acc,time.time() - since_batch))

            # Check if this is the best model wrt previous epochs
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # if persist_model != False:
                #     torch.save(model, '{}/best_{:.4f}acc_{}epochs_{}qubits.h5'.format(torch_dir + mdl_name, epoch_acc,
                #                                                                       num_epochs, model.fc.n_qubits))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
            if phase == 'train' and epoch_acc > best_acc_train:
                best_acc_train = epoch_acc
            if phase == 'train' and epoch_loss < best_loss_train:
                best_loss_train = epoch_loss

                # Print final results
    print('Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f} '.format('train' if phase == 'train' else 'val  ', epoch + 1,config.num_epochs, epoch_loss, epoch_acc))
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    # print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:.4f} | Best test accuracy: {:.4f}'.format(best_loss, best_acc))

    return model, best_loss, float(best_acc)


def train_classification_model(config, criterion, optimizer, scheduler, model):
    # Creating a folder to save the model performance.
    # try:
    #     os.mkdir(f'./modelPerformance/{name}')
    # except:
    #     print('Ignore directory creation failure.')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(config.num_epochs):
        # print('\n Epoch:{}/{}, MDL:{}, Q-block:{}, Q-bits:{}, Q-depth:{}'.format(epoch + 1, num_epochs, mdl_name, model.fc.q_net_block_name, model.fc.n_qubits,model.fc.q_depth))
        # print('-' * 100)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # epochs
            epoch = int(len(config.image_datasets[phase]) / config.batch_size)
            for _ in tqdm(range(epoch)):
                # Loading Data
                inputs, labels = next(iter(config.dataloaders[phase]))
                inputs = inputs.to(config.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.to(config.device)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / config.dataset_sizes[phase]
            epoch_acc = running_corrects.double() / config.dataset_sizes[phase]
            # AUC: {:.4f} , epoch_auc
            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # print('Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}'.format(phase, epoch + 1, num_epochs, it + 1, n_batches + 1, time.time() - since_batch), end='\r', flush=True)
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model,
                #            './modelPerformance/{}/best_model_{:.4f}acc_{}epochs.h5'.format(name, epoch_acc, num_epochs))

                train_losses = []
                valid_losses = []

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # with open(f'./modelPerformance/{name}/' + sorted(os.listdir(f'./modelPerformance/{name}/'))[-1], 'rb') as f:
    #     buffer = io.BytesIO(f.read())
    # model = torch.load(buffer)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss, best_acc
