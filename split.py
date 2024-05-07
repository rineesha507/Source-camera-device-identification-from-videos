import os # For interacting with the operating system
import shutil # For high-level file operations like copying files
import random  # For shuffling the list of images


def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1): #Ratios specifying how to split the dataset. Default values split the dataset into 80% training, 10% validation, and 10% testing sets
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')

    # The exist_ok=True argument ensures that the function doesn't raise an error if the directories already exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of class directories  inside the input_dir
    class_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))] #d'' represents each item

    # Split each class into train, val, and test
    for class_dir in class_dirs:
        class_input_dir = os.path.join(input_dir, class_dir)
        class_train_dir = os.path.join(train_dir, class_dir)
        class_val_dir = os.path.join(val_dir, class_dir)
        class_test_dir = os.path.join(test_dir, class_dir)

        os.makedirs(class_train_dir, exist_ok=True)
        os.makedirs(class_val_dir, exist_ok=True)
        os.makedirs(class_test_dir, exist_ok=True)

        # List images in class directory
        images = os.listdir(class_input_dir)
        random.shuffle(images)

        # Calculate split sizes
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        num_test = num_images - num_train - num_val

        # Split images into train, val, and test sets
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # Copy images to respective directories
        for img in train_images:
            shutil.copyfile(os.path.join(class_input_dir, img), os.path.join(class_train_dir, img))
        for img in val_images:
            shutil.copyfile(os.path.join(class_input_dir, img), os.path.join(class_val_dir, img))
        for img in test_images:
            shutil.copyfile(os.path.join(class_input_dir, img), os.path.join(class_test_dir, img))


input_dir = 'directory'
output_dir = 'split_directory'
split_dataset(input_dir, output_dir)
