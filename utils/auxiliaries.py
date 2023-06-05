import torch
import torchvision

import torch.optim as optim
from lion_pytorch import Lion
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm

import pandas as pd
import numpy as np
import json


from utils import constants as cst


def get_dataset():
    # We use the Cifar100 dataset
    dataset = torchvision.datasets.CIFAR100
    # Normalize the data
    data_mean = cst.DATA_MEAN
    data_stddev = cst.DATA_STD

    transform_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    transform_test = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(data_mean, data_stddev),
        ]
    )

    training_set = dataset(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_set = dataset(
        root="./data", train=False, download=True, transform=transform_test
    )

    # training_loader = torch.utils.data.DataLoader(
    #    training_set,
    #    batch_size=cst.TRAIN_BATCH_SIZE,
    #    shuffle=True,
    #    num_workers=cst.NUM_WORKERS,
    # )
    # test_loader = torch.utils.data.DataLoader(
    #    test_set,
    #    batch_size=cst.TEST_BATCH_SIZE,
    #    shuffle=False,
    #    num_workers=cst.NUM_WORKERS,
    # )

    return training_set, test_set


def get_folds(k_folds=5):
    # Retrieve the full Training Dataset
    training_set, test_set = get_dataset()

    # Define the K-fold Cross Validator
    fold = KFold(n_splits=k_folds)

    # K-fold Cross Validation model evaluation
    folds = []
    ##### FOLDING  #####
    for fold, (train_ids, val_ids) in enumerate(fold.split(training_set)):
        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        # Define data loaders for training and testing data in this fold
        training_loader = torch.utils.data.DataLoader(
            training_set, batch_size=cst.TRAIN_BATCH_SIZE, sampler=train_subsampler
        )
        validation_loader = torch.utils.data.DataLoader(
            training_set, batch_size=cst.TEST_BATCH_SIZE, sampler=test_subsampler
        )

        folds.append((training_loader, validation_loader))

    return folds


# Optmizer we can use:  SGD, SGDM, RMS, ADAM, LION
def get_optimizer(optimizer_name, hyperparams, model_parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """

    #if optimizer_name == "LION":
    #    optimizer = Lion(
    #        model_parameters,
    #        lr=hyperparams["lr"],
    #        weight_decay=hyperparams["weight_decay"],
    #        momentum=hyperparams["momentum"]
    #    )
    #    
    #else:
    #    raise ValueError("Unexpected value for optimizer")
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters, lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        ) 
    
    elif optimizer_name == "SGDM":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            momentum=hyperparams["momentum"]
        )
    elif optimizer_name == "RMS":
        optimizer = torch.optim.RMSprop(
            model_parameters,
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
            momentum=hyperparams["momentum"]
        )
    elif optimizer_name == "ADAM":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
    elif optimizer_name == "LION":
        optimizer = Lion(
            model_parameters,
            lr=hyperparams["lr"],
            weight_decay=hyperparams["weight_decay"],
        )
        
    else:
        raise ValueError("Unexpected value for optimizer")
    return optimizer


def get_model(config, device):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """
    if config["model"] == "resnet18":
        model = torchvision.models.resnet18(num_classes=cst.NUM_CLASSES)
    elif config["model"] == "resnet101":
        model = torchvision.models.resnet101(num_classes=cst.NUM_CLASSES)
    else:
        raise ValueError("Unexpected model")

    model.to(device)
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True
    return model

def get_hyperparams(model_name, optimizer_name, sheet_path):
    # Load the Hyperparams sheet
    hyperparams_df = pd.read_excel(sheet_path)

    # Rename the columns
    hyperparams_df.rename(columns={
        "Model" : "model",
        "Optimizer" : "optimizer_name",
        "Learning Rate" : "lr",
        "Weight Decay" : "weight_decay",
        "Momentum" : "mom",
        "Accuracy" : "accuracy"
    }, inplace=True)


    # Focus on the relavant line : correct model and optimizer
    hyperparams_df_relevant = hyperparams_df[(hyperparams_df["model"] == model_name) &
                                         (hyperparams_df["optimizer_name"] == optimizer_name)].reset_index()

    # From a list of written numbers (strings) to a list of floats
    hyperparams_df_relevant.loc[:, 'lr'] = hyperparams_df_relevant["lr"].apply(lambda numbers : [float(nb) for nb in numbers.split(",")] if  not(isinstance(numbers, float)) else numbers)
    hyperparams_df_relevant.loc[:, 'weight_decay'] = hyperparams_df_relevant["weight_decay"].apply(lambda numbers : [float(nb) for nb in numbers.split(",")] if not(isinstance(numbers, float)) else numbers)
    hyperparams_df_relevant.loc[:, 'mom'] = hyperparams_df_relevant["mom"].apply(lambda numbers : [float(nb) for nb in numbers.split(",")] if not(isinstance(numbers, float)) else numbers)
    # Convert empty entries to empty lists
    #hyperparams_df_relevant.fillna([], inplace=True)

    # Finally, build the hyperparams dict
    hyperparams = {
        "lr" : hyperparams_df_relevant["lr"][0],
        "weight_decay" : hyperparams_df_relevant["weight_decay"][0],
        "momentum" : hyperparams_df_relevant["mom"][0],
    }
    return hyperparams

def store_grid_search_results(all_accuracies, all_hyperparams, best_accuracy, best_hyperparams, model_name, optimizer_name, sheet_path):
    # Load the Hyperparams sheet
    hyperparams_df = pd.read_excel(sheet_path)

    model_mask = hyperparams_df["Model"] == model_name
    opti_mask = hyperparams_df["Optimizer"] == optimizer_name

    #assert len(all_accuracies) == len(all_hyperparams)
    #results = [(all_hyperparams[k], all_accuracies[k]) for k in range(len(all_accuracies))]

    hyperparams_df.loc[model_mask & opti_mask, "Best Accuracy"] = best_accuracy
    hyperparams_df.loc[model_mask & opti_mask, "Best Hyperparamaters"] = str(best_hyperparams)
    #hyperparams_df.loc[model_mask & opti_mask, "Accuracies"] = all_accuracies

    hyperparams_df.to_excel(sheet_path, index=False)



def train_model(
    model,
    optimizer,
    training_loader,
):
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    loss_list = []
    trained_examples = []

    for epoch in range(0, cst.EPOCHS):
        # one training iteration
        loss_total = 0
        loss_count = 0
        print(f"Epoch: {epoch+1}|{cst.EPOCHS}")
        batch = 0
        for data, label in tqdm(training_loader, desc="Training"):
            batch += 1

            optimizer.zero_grad()

            data = data.to(cst.DEVICE)
            label = label.to(cst.DEVICE)

            pred = model(data)

            loss = criterion(pred, label)
            loss_total += loss.item()
            loss_count += 1

            loss.backward()
            optimizer.step()

            if batch % cst.PLOT_GRANULARITY == 0:
                # append the average of all losses since the last recorded one
                loss_average = loss_total / loss_count
                loss_list.append(loss_average)
                # reset variables once we record
                loss_total = 0
                loss_count = 0
                trained_examples.append(
                    cst.TRAIN_BATCH_SIZE * batch + epoch * cst.TRAINING_SET_SIZE
                )
    # torch.save(model.state_dict(), cst.MODEL_STORE_DIR + store_name)

    return loss_list, trained_examples, model


def test_model(model, test_loader):
    model.eval()
    total_loss = 0
    total_correct = 0

    criterion = torch.nn.CrossEntropyLoss()
    test_dataset_size = 0
    nb_batch = 0
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc="Testing"):
            data = data.to(cst.DEVICE)
            label = label.to(cst.DEVICE)

            test_dataset_size += data.size()[0]
            nb_batch += 1

            preds = model(data)
            loss = criterion(preds, label)
            total_loss += loss.item()
            pred = preds.data.max(1, keepdim=True)[1]
            correct = pred.eq(label.data.view_as(pred)).sum()
            total_correct += correct

    average_loss = total_loss / nb_batch
    accuracy = total_correct / test_dataset_size

    return average_loss, accuracy


def grid_search(model_name, optimizer_name, hyperparams, folds, save_path):
    num_folds = len(folds)
    all_accuracies = []
    all_hyperparams = []
    best_accuracy = 0
    best_hyperparams = None

    for lr in hyperparams["lr"]:
        for weight_decay in hyperparams["weight_decay"]:
            for mom in hyperparams["momentum"]:
                current_hyperparams = {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "momentum": mom,
                }
                config = dict(
                    model=model_name,
                    optimizer=optimizer_name,
                    hyperparams=hyperparams,
                )

                # Model and Optimizer Initialization
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model = get_model(config, cst.DEVICE)

                optimizer = get_optimizer(optimizer_name, current_hyperparams, model.parameters())

                current_hyperparams_accuracy = 0

                print(f"--- Working with {optimizer_name} on {model_name} ---")

                print("Hyperparameters: \n" + \
                  f"Learning Rate : {lr} \n" + \
                  f"Weight Decay : {weight_decay} \n" + \
                  f"Momentum : {mom}"
                )

                # Running K-Fold cross-validation training
                fold_nb = 0
                for training_loader, validation_loader in folds:

                    fold_nb += 1
                    print(f"----- Training on fold # {fold_nb} -----")
                    (
                        training_loss,
                        trained_examples,
                        model
                    ) = train_model(
                        model=model,
                        optimizer=optimizer,
                        training_loader=training_loader,
                    )
                    representative_training_loss = training_loss[-1]
                    print(f"Average lost on the last bacthes : {representative_training_loss}")
                    # Running Validation
                    (average_validation_loss, 
                    validation_accuracy) = test_model(
                        model=model, 
                        test_loader=validation_loader
                    )
                    print(f"Accuracy : {validation_accuracy}")
                    current_hyperparams_accuracy += validation_accuracy 
                    
                    # K-Fold is too heavy, we run just 1 fold
                    break
                    # Keep track of the accuracies achieved
                #current_hyperparams_accuracy /= num_folds


                all_accuracies.append(current_hyperparams_accuracy)
                all_hyperparams.append([lr, weight_decay, mom])

                if current_hyperparams_accuracy > best_accuracy:
                    best_accuracy = current_hyperparams_accuracy
                    best_hyperparams = [lr, weight_decay, mom]
                    torch.save(model.state_dict(), save_path)
                    print("\n Best Hyperparameters so far, Model Saved !")
                    print(f"Best Accuracy : {best_accuracy}")

    return all_accuracies, all_hyperparams, best_accuracy, best_hyperparams


def train_with_fold(model_name, optimizer_name, hyperparams, folds):
    num_folds = len(folds)
    accuracy = 0

    config = dict(
        model=model_name,
        optimizer=optimizer_name,
        hyperparams=hyperparams,
    )

    model = get_model(config, cst.DEVICE)
    optimizer = get_optimizer(optimizer_name, hyperparams, model.parameters())

    training_loss_full = []

    # Running K-Fold cross-validation training
    fold_nb = 0
    for training_loader, validation_loader in folds:

        fold_nb += 1
        print(f"----- Training on fold # {fold_nb} -----")
        (
            training_loss,
            trained_examples,
            model
        ) = train_model(
            model=model,
            optimizer=optimizer,
            training_loader=training_loader,
        )
        training_loss_full.append(training_loss)
        print(f"Training Loss: {training_loss}")


        # Running Validation
        (average_validation_loss, 
        validation_accuracy) = test_model(
            model=model, 
            test_loader=validation_loader
        )
        print(f"Fold {fold_nb} Accuracy : {validation_accuracy}")
        accuracy += validation_accuracy 

    accuracy /= num_folds
    print(f"Full Accuracy: {accuracy}")

    np.sum(training_loss_full, axis=0)
    training_loss_full = [x / num_folds for x in training_loss_full]
    print(f"Training Loss Array: {training_loss_full}")


    return accuracy, training_loss_full

def train(model_name, optimizer_name, hyperparams, training_loader, test_loader):

    config = dict(
        model=model_name,
        optimizer=optimizer_name,
        hyperparams=hyperparams,
    )

    model = get_model(config, cst.DEVICE)
    optimizer = get_optimizer(optimizer_name, hyperparams, model.parameters())

    (
        training_loss,
        trained_examples,
        model
    ) = train_model(
        model=model,
        optimizer=optimizer,
        training_loader=training_loader,
    )
    print(f"Training Loss: {training_loss}")

    (test_loss, 
    acc) = test_model(
        model=model, 
        test_loader=test_loader
    )

    print(f"Training Loss Array: {training_loss}")
    print(f"Test Accuracy: {acc}")
    print(f"Test Loss Array: {test_loss}")

    return training_loss, acc, test_loss, trained_examples


def write_to_file(optimizer_name, name, arr):
    with open(f'{name}-{optimizer_name}.txt', 'w') as filehandle:
        for a in arr:
            filehandle.write(f'{a}\n')

def read_from_file(optimizer_name, name):
    list_ = []

    with open(f'{name}-{optimizer_name}.txt', 'r') as filehandle:
        for line in filehandle:
            curr_place = line[:-1]
            list_.append(float(curr_place))
    return list_

def store_results(train_loss, acc, test_loss, trained_examples, filename, path):
    store = {
        'train_loss' : train_loss,
        'test_loss' : test_loss,
        'acc' : acc,
        'trained_examples' : trained_examples
    }
    json.dump(store, open(path + f"/{filename}.txt", "w"))


def load_results(filename, path):
    return json.load(open(path + f"/{filename}.txt"))