from torchvision import datasets
from random import random
from .visda_rotator import VisdaRotator

class VisdaDataset(datasets.ImageFolder):
  def set_params(self,rot_list=[-2,-1,0,1,2], rot_weights=[1,3,0,3,1], light_list = [-1,0,1], light_weights = [1,1,1],n_views=2):
    self.n_views = n_views
    self.rot_list = rot_list
    self.rot_weights = rot_weights
    self.light_list = light_list
    self.light_weights = light_weights
    self.rotator = VisdaRotator(self.root)

  def get_random_rots(self):
    return [0] + random.choices(self.rot_list, self.rot_weights,k=self.n_views-1)

  def get_random_lights(self):
    return [0] + random.choices(self.light_list, self.light_weights, k=self.n_views-1)

  def __getitem__(self, index):
    path, target = self.samples[index]
    rot_list = self.get_random_rots()
    light_list = self.get_random_lights()
    short_path = path[len(str(self.root))+1:]
    path_list = [path] + [self.root/self.rotator.next_image(short_path,rot_list[i], light_list[i]) for i in range(1,self.n_views)]
    sample_list = [self.loader(path) for path in path_list]
   
    if self.transform is not None:
        sample_list = [self.transform(sample) for sample in sample_list]
   
    if self.target_transform is not None:
        target = self.target_transform(target)

    return sample_list, target