# Dataset Structure

The project data needs to be structured as specified in this document.

All the data should be placed within the `data` folder.
Each dataset should be placed in a subfolder with a unique name - the default is 'Sint-Maarten-2017'.
Each dataset should have three subfolders within,
    - 'train' containing the images from which the model will learn to identify damage (typically 80% of all labelled data)
    - 'validation' containing the images which will be used to tune the model (typically 10% of all labelled data)
    - 'test' containing the images which will be used to score the model (typically 10% of all labelled data)

Each subfolder is split into two folders,
    - 'before' containing the images of the region before the disaster
    - 'after' containing the images of the region after the disaster
    - The two folders should contain the exact same number of files with the exact same filenames
    - A text file (labels.txt) containing the unique filename (before and after images share this name) and the level of damage of the building in this image
