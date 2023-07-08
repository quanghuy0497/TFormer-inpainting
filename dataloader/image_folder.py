import os
import os.path
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    """
    dir: directory paths that store the image with the format:
        directory/
        ├── class_1
        │   ├── img_1.jpg
        │   ├── img_2.jpg
        │   ├── ...
        │   └── img-n.jpg
        ├── class_2
        │    ├── img_1.jpg
        │    ├── img_2.jpg
        │    ├── ...
        │    └── img_n.jpg
        └── ...        
    return: image paths, label, and sizes of images
    """
    
    img_paths = []
    labels = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for label in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir, label)):
            for fname in sorted(os.listdir(os.path.join(dir, label))):
                if is_image_file(fname):
                    path = os.path.join(dir, label, fname)
                    labels.append(label)
                    img_paths.append(path)

    return img_paths, labels, len(img_paths)
