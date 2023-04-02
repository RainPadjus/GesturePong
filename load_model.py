import os
from fastai.vision.all import *
from fastcore.all import *

data_path = Path("data")

def load_images_from_folder(folder_path):
    return get_image_files(folder_path)

def create_label_from_folder(folder_path):
    return folder_path.name

def load_labeled_images(data_path):
    labeled_images = []
    for folder in data_path.iterdir():
        if folder.is_dir():
            label = create_label_from_folder(folder)
            images = load_images_from_folder(folder)
            for img in images:
                labeled_images.append((img, label))
    return labeled_images

def train_model():
    labeled_images = load_labeled_images(data_path)

    print(labeled_images[:5])  # Print the first 5 labeled images
    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=[Resize(192, method='squish')]
    ).dataloaders(data_path, bs=32)

    dls.show_batch(max_n=6)

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(1) #model used was finetuned 10 cycles
    learn.save("my_model")
    learn.export("trained_model")

if __name__ == "__main__":
    train_model()
