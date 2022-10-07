from .utils import MultiLevelDict
import pandas as pd
import numpy as np

class VisdaManager(object):
  def __init__(self, root_path):
    images_data_file = root_path / 'image_list_with_data.csv'
    if images_data_file.exists():
      self._images_data = pd.read_csv(images_data_file)
    else:
      image_list = pd.read_csv(root_path/"image_list.txt", sep=' ',names=["path", "target"])
      self._images_data = self._generate_images_data(image_list)
      self._images_data.to_csv(root_path/"image_list_with_data.csv", index=False)
    self._path_dict = MultiLevelDict()
    for record in self._images_data.to_dict('records'):
      keys = [record[column] for column in ['directory', 'src', 'object_id', 'cam_yaw', 'light_yaw']]
      path = record['path']
      self._path_dict.add(keys, path, record["target"])
    # self._all_cam_yaw = self._images_data["cam_yaw"].value_counts().keys().to_list()
    # self._all_light_yaw = self._images_data["light_yaw"].value_counts().keys().to_list()
    
  def _generate_images_data(self, image_list):
    temp_data = image_list.copy()
    temp_data[['directory', 'src', 'object_id', 'cam_yaw', 'light_yaw', 'cam_pitch']] = temp_data.apply(lambda row: pd.Series([x for x in self._extract_params(row["path"])]), axis=1)
    return temp_data
  
  def _extract_params(self, image_path):
    directory, image_name = image_path.split(r"/")
    src = image_name[:5]
    object_id, others = image_name[6:].split("__")
    [cam_yaw, light_yaw, cam_pitch] =[x for x in others.split("_") if x != ""]
    cam_pitch = cam_pitch.split(".")[0]
    return directory, src, object_id, int(cam_yaw), int(light_yaw), int(cam_pitch)

  @property
  def object_count(self):
    return len(self._path_dict.id_to_adress)

  def get_object_path_list(self, index):
    keys = self._path_dict.id_to_adress[index]
    return self._path_dict.flatten(keys)

  def get_label(self, index):
    return self._path_dict.id_to_target[index]

  # def next_image(self, image_path, rot=1, light_rot=0):
  #   directory, src, object_id, cam_yaw, light_yaw, _ = self._extract_params(image_path)
  #   try:
  #     result = self._get_image_path(directory, src, object_id, self._next_cam_yaw(cam_yaw,rot),self._next_light_yaw(light_yaw,light_rot))
  #   except KeyError:
  #     next_cam_yaw = self._next_partial_cam_yaw(directory, src, object_id, cam_yaw, rot)
  #     light_yaw = self._next_partial_light_yaw(directory, src, object_id, next_cam_yaw, light_yaw, light_rot)
  #     result = self._get_image_path(directory, src, object_id, next_cam_yaw, light_yaw)
  #   return result

  # def _next_partial_cam_yaw(self, directory, src, object_id, cam_yaw, n):
  #   available = self._path_dict.get([directory, src, object_id])
  #   available_cam_yaws = list(available.keys())
  #   index = available_cam_yaws.index(cam_yaw)
  #   return available_cam_yaws[(index + n)%len(available_cam_yaws)]

  # def _next_partial_light_yaw(self, directory, src, object_id, cam_yaw, light_yaw, n):
  #   available = self._path_dict.get([directory, src, object_id, cam_yaw])
  #   available_light_yaws = list(available.keys())
  #   try:
  #     index = available_light_yaws.index(light_yaw)
  #   except ValueError:
  #     index = np.argmin([abs(x - light_yaw) for x in available_light_yaws])
  #   return available_light_yaws[(index+n)%len(available_light_yaws)]

  # def _next_cam_yaw(self, cam_yaw, n=1):
  #   index = self._all_cam_yaw.index(cam_yaw)
  #   return self._all_cam_yaw[(index + n)%len(self._all_cam_yaw)]

  # def _next_light_yaw(self, light_yaw, n=1):
  #   index = self._all_light_yaw.index(light_yaw)
  #   return self._all_light_yaw[(index + n)%len(self._all_light_yaw)]


  # def _get_image_path(self,directory, src, object_id, cam_yaw, light_yaw):
  #   return self._path_dict.get([directory, src, object_id, cam_yaw, light_yaw])