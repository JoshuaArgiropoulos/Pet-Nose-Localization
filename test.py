import argparse
import os
import math
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import CustomNet as CustomNet
import CustomDataset as CustomDataset
import time

def eval_acc_for_epoch(data, CustomNet, device):
    print("Statistics for Test Dataset:")

    CustomNet.eval() 
    
    total_time = 0
    total_images = 0
    distances = []
    for _, (imgs, labels) in enumerate(data):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            start_time = time.time()  # Start time measurement
            output = CustomNet(imgs)
            
            end_time = time.time()  # End time measurement
            total_time += (end_time - start_time)
            total_images += imgs.size(0)

        for i in range(imgs.size(0)):
            img = imgs[i].cpu().numpy().transpose((1, 2, 0))

            scaledImg = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)

            predicted = output[i].cpu().numpy()
            confirmed = labels[i].cpu().numpy()

            predicted_x = int(predicted[0] * 300)
            predicted_y = int(predicted[1] * 300)
            target_x = int(confirmed[0] * 300)
            target_y = int(confirmed[1] * 300)

            target_xy = (target_x, target_y)
            predicted_xy = (predicted_x, predicted_y)

            euclidean_dist = math.sqrt(
                (predicted_x - target_x) ** 2 + (predicted_y - target_y) ** 2
            )

            distances.append(euclidean_dist)

            max_distance = max(distances)
            min_distance = min(distances)
            mean_distance = sum(distances) / len(distances)
            std_distance = torch.tensor(distances).std().item()

            # ---------------------------------------------- Uncomment to show images ----------------------------------------------#
            # cv2.circle(scaledImg, target_xy, 2, (0, 0, 255), 1)
            # cv2.circle(scaledImg, predicted_xy, 2, (0, 255, 0), 1)
            # cv2.imshow("boxes", scaledImg)
            key = cv2.waitKey(0)
            if key == ord("x"):
                break
    avg_time_per_image = total_time / total_images
    print(f"Average inference time per image: {avg_time_per_image * 1000:.2f} milliseconds")


    print("Minimum Distance: {}".format(min_distance))
    print("Maximum Distance: {}".format(max_distance))
    print("Mean Distance: {}".format(mean_distance))
    print("Standard Deviation of Distance: {}".format(std_distance))


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--dir", type=str, help="Images Directory")
    parser.add_argument("-b", "--b", type=int, default=48, help="Batch Size")
    parser.add_argument("-l", "--l", default="./model.pth", help="model.pth file")
    parser.add_argument("-cuda", "--cuda", default="Y")

    args = parser.parse_args()

    image_dir = os.path.abspath(args.dir)

    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((300, 300), antialias=True)]
    )
    test_data = CustomDataset.CustomDataset(
        image_dir, is_train=False, transform=train_transform
    )

    test_data = DataLoader(test_data, batch_size=args.b, shuffle=False)
    encoder = CustomNet.encoder_decoder.encoder
    CustomNet = CustomNet.CustomNet(encoder)

    if args.l is not None:
        CustomNet.load_state_dict(torch.load(args.l))

    # Set the device (GPU if available, otherwise CPU)
    if torch.cuda.is_available() and args.cuda.lower() == "y":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device: {}\n".format(device))

    CustomNet.to(device)

    eval_acc_for_epoch(test_data, CustomNet, device)
