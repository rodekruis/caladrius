# Dataset Structure

The project data needs to be *structured* as specified in this document.

All the *data* should be placed within the `data` folder. (i.e. `./data`)

Each **dataset** should be placed in a subfolder under `data` with a unique name. (e.g. `./data/Sint-Maarten-2017`)

Each *dataset* should have **four subfolders** within,

- `train` containing the images from which the model will *learn* to identify damage. (typically 80% of all labelled data) (i.e. `./data/Sint-Maarten-2017/train`)
- `validation` containing the images which will be used to *tune* the model. (typically 10% of all labelled data) (i.e. `./data/Sint-Maarten-2017/validation`)
- `test` containing the images which will be used to *score* the model. (typically 10% of all labelled data) (i.e. `./data/Sint-Maarten-2017/test`)
- `inference` containing the images for which the model will be used on. (typically all unlabelled data) (i.e. `./data/Sint-Maarten-2017/inference`)

Each *subfolder* is split into **two folders** and **one file**,

- `before` containing the images of the region **before the disaster**. (i.e. `./data/Sint-Maarten-2017/test/before`)
- `after` containing the images of the region **after the disaster**. (i.e. `./data/Sint-Maarten-2017/test/after`)
- The `before` and `after` folders should contain the *matching pairs* files with the *same filename*. (e.g. In `Sint-Maarten-2017` the building `OBJECTID` is used as the filename) (i.e. `./data/Sint-Maarten-2017/test/before/10134.png` and `./data/Sint-Maarten-2017/test/after/10134.png`)
- *Exempted for Inference Set* - A text file (`labels.txt`) with each line containing the filename (before and after images share this name) and the level of damage of the building in this image (i.e. `10134.png 0.6030`)
