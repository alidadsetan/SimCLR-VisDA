from pathlib import Path
from torchvision.datasets.folder import default_loader, ImageFolder
import random
from .visda_rotator import VisdaManager

class VisdaUnsupervisedDataset(object): 
  '''
  Combines train and vali datasets. but because train dataset is so small, 
  it will repeat the train dataset, train_repetition times.
  '''
  def __init__(self, train_dataset, valid_dataset, train_repetition=15):
    self.train = train_dataset
    self.valid = valid_dataset
    self.train_repetition = train_repetition

  def __len__(self):
    return len(self.train) * self.train_repetition + len(self.valid)

  def __getitem__(self, index):
    if index < len(self.train) * self.train_repetition:
      return self.train[index%len(self.train)]
    return self.valid[index - len(self.train) * self.train_repetition]


class VisdaValidDataset(ImageFolder):
  def __init__(self, *args, **kwargs):
    if 'n_views' in kwargs:
      n_views = kwargs.pop('n_views')
    else:
      n_views = args.pop()
    super().__init__(*args,**kwargs)
    self.n_views = n_views 

  def set_param(self, n_views: int) -> None: 
    self.n_views = n_views

  def __getitem__(self, index):
    path, target = self.samples[index]
    loaded = self.loader(path)

    result = []
    if self.transform is not None:
      result = [self.transform(loaded) for _ in range(self.n_views)]

    targets = []
    if self.target_transform is not None:
      targets = [self.target_transform(target) for _ in range(self.n_views)]
    else:
      targets = [-1 for _ in range(self.n_views)]

    return result, targets

class VisdaTrainDataset(object):
  def __init__(self, root: Path, transform=None, n_views=2):
    self.root = root
    self.visda_manager = VisdaManager(root)
    self.transform = transform
    self.n_views = n_views

  def __len__(self):
    return self.visda_manager.object_count

  def __getitem__(self, index):
    path_list = self.visda_manager.get_object_path_list(index) # list of relative pathes for different images of object with index index
    selected_path_list = random.choices(path_list,k=self.n_views)
    result = [default_loader(self.root/path) for path in selected_path_list]

    if self.transform is not None:
      result = [self.transform(x) for x in result]
    
    label = self.visda_manager.get_label(index)
    return result, [label for _ in result]
