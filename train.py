import argparse
import os
import datetime
import time as time_module

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import CustomNet as CustomNet
import CustomDataset as custom_dataset


def calculate_euclidean_distance(predictions, actuals, img_width=300, img_height=300):
    distance = torch.sqrt(torch.sum((actuals - predictions) ** 2, axis=1))
    diagonal_length = torch.sqrt(torch.tensor(img_width**2 + img_height**2))
    return (distance / diagonal_length) * 100


def compute_accuracy(loader, neural_model, device, is_test=False):
    cumulative_distance = 0.0
    count_images = 0
    data_type = "Test Data" if is_test else "Training Data"

    neural_model.eval()
    with torch.no_grad():
        for index, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            predicted_labels = neural_model(images)
            cumulative_distance += calculate_euclidean_distance(
                predicted_labels, labels
            )
            count_images += labels.size(0)
            print(f"{data_type} Accuracy Progress: {index}/{len(loader)}")

    neural_model.train()
    return (cumulative_distance / count_images) * 100


def execute_training_epoch(
    epoch_count,
    optimizer,
    neural_model,
    loss_function,
    data_loader,
    lr_scheduler,
    compute_device,
    model_save_path=None,
    plot_path=None,
    eval_epochs=False,
    output_folder="./",
    test_loader=None,
):
    neural_model.train()
    total_train_losses, total_test_losses = [], []
    train_accuracy, test_accuracy = [], []

    for epoch in range(1, epoch_count + 1):
        epoch_start_time = time_module.time()
        current_time = datetime.datetime.now().strftime("%I:%M:%S %p on %B %d, %Y")
        print(f"\nStarting Epoch {epoch} @ {current_time}")

        # Training phase
        cumulative_loss = 0.0
        for batch_idx, (images, labels) in enumerate(data_loader):
            batch_start_time = time_module.time()
            images, labels = images.to(compute_device), labels.to(compute_device)
            optimizer.zero_grad()

            predictions = neural_model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.item()
            batch_duration = time_module.time() - batch_start_time
            print(
                f"Processing Batch {batch_idx + 1}/{len(data_loader)} - Duration: {batch_duration:.2f}s"
            )

        # Scheduler step and loss calculation
        lr_scheduler.step(cumulative_loss)
        epoch_loss = cumulative_loss / len(data_loader)
        total_train_losses.append(epoch_loss)

        # Model saving
        if model_save_path and (epoch % 20 == 0 or epoch == epoch_count):
            save_path = os.path.join(os.path.abspath(output_folder), model_save_path)
            torch.save(neural_model.state_dict(), save_path)
            print(f"Model saved as {model_save_path}")

        # Test evaluation
        if test_loader:
            neural_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(compute_device), labels.to(
                        compute_device
                    )
                    test_predictions = neural_model(images)
                    loss = loss_function(test_predictions, labels)
                    test_loss += loss.item()
            neural_model.train()

            avg_test_loss = test_loss / len(test_loader)
            total_test_losses.append(avg_test_loss)

        # Accuracy evaluation
        if eval_epochs:
            train_acc = compute_accuracy(data_loader, neural_model, compute_device)
            test_acc = (
                compute_accuracy(
                    test_loader, neural_model, compute_device, is_test=True
                )
                if test_loader
                else 0
            )
            train_accuracy.append(train_acc)
            test_accuracy.append(test_acc)
            print(
                f"Training Data Accuracy: {train_acc}% | Test Data Accuracy: {test_acc}%"
            )

        # Plotting
        if plot_path:
            plot_save_path = os.path.join(os.path.abspath(output_folder), plot_path)
            plt.figure(figsize=(12, 6))
            plt.plot(total_train_losses, label="Training Loss")
            plt.plot(total_test_losses, label="Test Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training and Test Loss Over Epochs")
            plt.legend()
            plt.savefig(plot_save_path)

        # Epoch summary
        epoch_duration = time_module.time() - epoch_start_time
        print(
            f"Epoch {epoch} - Training Loss: {epoch_loss:.4f} | Duration: {epoch_duration:.2f}s"
        )

    return total_train_losses, total_test_losses


# Main execution
if __name__ == "__main__":
    # Argument parsing
    arg_parser = argparse.ArgumentParser()
    # Dataset directory argument
    arg_parser.add_argument(
        "-dir", "--dir", type=str, required=True, help="Directory path for images"
    )
    # Training arguments
    arg_parser.add_argument("-e", "--e", type=int, default=40, help="Number of epochs")
    arg_parser.add_argument("-b", "--b", type=int, default=48, help="Batch size")
    arg_parser.add_argument(
        "-l", "--l", required=False, help="Load saved model file path"
    )
    arg_parser.add_argument(
        "-s", "--s", default="model.pth", help="Save model file name"
    )
    arg_parser.add_argument(
        "-p", "--p", default="loss_plot.png", help="Loss plot file name"
    )
    arg_parser.add_argument("-cuda", "--cuda", default="Y")

    # Hyperparameter arguments
    arg_parser.add_argument("-lr", "--lr", type=float, default=0.001)
    arg_parser.add_argument("-wd", "--wd", type=float, default=0.00001)
    arg_parser.add_argument("-minlr", "--minlr", type=float, default=0.001)
    arg_parser.add_argument("-gamma", "--gamma", type=float, default=0.9)
    
    args = arg_parser.parse_args()


    # Dataset and model setup
    dataset_directory = os.path.abspath(args.dir)
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((300, 300), antialias=True)]
    )

    # Training and test datasets
    training_dataset = custom_dataset.CustomDataset(
        dataset_directory, is_train=True, transform=data_transform
    )
    testing_dataset = custom_dataset.CustomDataset(
        dataset_directory, is_train=False, transform=data_transform
    )

    train_loader = DataLoader(training_dataset, batch_size=args.b, shuffle=True)
    test_loader = DataLoader(testing_dataset, batch_size=args.b, shuffle=False)

    # Model initialization
    model_encoder = CustomNet.encoder_decoder.encoder
    current_model = CustomNet.CustomNet(model_encoder)

    if args.l:
        current_model.load_state_dict(torch.load(args.l))

    # Device setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda == "Y" else "cpu"
    )
    print(f"Using device: {device}")

    # Training components
    loss_function = nn.MSELoss().to(device)
    current_model.to(device)
    optimizer = optim.Adam(current_model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=2,
        verbose=True,
        factor=args.gamma,
        min_lr=args.minlr,
    )

    # Execute training
    execute_training_epoch(
        args.e,
        optimizer,
        current_model,
        loss_function,
        train_loader,
        lr_scheduler,
        device,
        model_save_path=args.s,
        plot_path=args.p,
        eval_epochs=True,
        output_folder="./",
        test_loader=test_loader,
    )
