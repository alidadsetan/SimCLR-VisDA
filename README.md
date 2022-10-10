# SimCLR-VisDa

## Motivation
Using SimCLR algorithm for VisDa2017 challenge of unsupervised domain adaptation. We will use the standard SimCLR algorithm, with the addition of considering rotations (available in VisDa dataset) as a transformation in the algorithm.

### Initial hope?
We hope to improve on unsupervised domain adaptation task on VisDa2017 challenge.

## requirements
Python 3.9.14

## Steps


### Cloning repo, Downloading and extracting dataset
*TODO*: download tar files, extract, ...

### Installing dependencies
```
pip install -r req.txt
```
### Unsupervised learning
As the first step, we would want to train our Resnet50 over our dataset. for this, run.
```bash
python run.py --action finetune --storage /path/to/your/extracted/tarfiles
```
This command will create a file named `image_list_with_data.csv` in train directory in ‍‍location you provided by `--storage`. This file is used for making faster executions on later experiments.
Also, after this command is 
 default for every parameter is (TODO) Caution: this would take ***a lot***.

### 