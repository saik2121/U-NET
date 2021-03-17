import albumentations as A
import torch.nn as nn
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from Model import Unet
from utils import (
    get_loaders,
    check_acc,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE=16
NUM_EPOCHS=10
NUM_WORKERS=2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader,model,optimiser,loss_fn):
    loop=tqdm(loader)

    for index,(data,targets) in enumerate(loop):
        data=data.to(DEVICE)
        targets=targets.float().unsqueeze(1).to(DEVICE)

        #FORWARD
        with torch.cuda.amp.autocast():
            preds=model(data)
            loss=loss_fn(preds,targets)

        #BACKWARD
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        loop.set_postfix(loss=loss.item())


def main():
    train_transform=A.compose(
        [
            A.HorizontalFlip(p=0.1),
            A.VerticalFlip(p=0.5),
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform=A.compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = Unet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader,val_loader=get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
    )
    
    check_acc(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for i in range(NUM_EPOCHS):
        train_fn(train_loader,model,optimiser,loss_fn)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)


        check_acc(val_loader, model, device=DEVICE)

        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
