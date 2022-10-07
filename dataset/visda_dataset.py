from pathlib import Path
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from random import random
from .visda_rotator import VisdaManager

class VisdaDataset(object): 
  '''
  Combines train and vali datasets. but because train dataset is so small, 
  it will repeat the train dataset, train_repetition times.
  '''
  def __init__(self, train_dataset, valid_dataset, train_repetition=10):
    self.train = train_dataset
    self.valid = valid_dataset
    self.train_repetition = train_repetition

  def __len__(self):
    return len(self.train) * self.train_repetition + len(self.valid)

  def __getitem__(self, index):
    if index < len(self.train) * self.train_repetition:
      return self.train[index%len(self.train)]
    return self.valid[index - len(self.train) * self.train_repetition]


class VisdaTrainDataset(object):
  def __init__(self, root: Path, transform=None, n_views=5):
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

# class LegacyVisdaDataset(datasets.ImageFolder):
#   def set_params(self,rot_list=[-2,-1,0,1,2], rot_weights=[1,3,0,3,1], light_list = [-1,0,1], light_weights = [1,1,1],n_views=2):
#     self.n_views = n_views
#     self.rot_list = rot_list
#     self.rot_weights = rot_weights
#     self.light_list = light_list
#     self.light_weights = light_weights
#     self.rotator = VisdaManager(self.root)

#   def get_random_rots(self):
#     return [0] + random.choices(self.rot_list, self.rot_weights,k=self.n_views-1)

#   def get_random_lights(self):
#     return [0] + random.choices(self.light_list, self.light_weights, k=self.n_views-1)

#   def __getitem__(self, index):
#     path, target = self.samples[index]
#     rot_list = self.get_random_rots()
#     light_list = self.get_random_lights()
#     short_path = path[len(str(self.root))+1:]
#     path_list = [path] + [self.root/self.rotator.next_image(short_path,rot_list[i], light_list[i]) for i in range(1,self.n_views)]
#     sample_list = [self.loader(path) for path in path_list]
   
#     if self.transform is not None:
#         sample_list = [self.transform(sample) for sample in sample_list]
   
#     if self.target_transform is not None:
#         target = self.target_transform(target)

#     return sample_list, target

