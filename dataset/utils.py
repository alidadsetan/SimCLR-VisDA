from typing import Dict
def _flatten(X):
  if isinstance(X, Dict):
    result = []
    for x in X.values():
      result += _flatten(x)
    return result
  return [X]

class MultiLevelDict(object):
  def __init__(self):
    self._root = {}
    self.id_to_adress = []
    self.id_to_targets = []

  def add(self, keys, value, target):
    where = self._root
    for index, key in enumerate(keys[:-1]):
      if not key in where:
        where[key] = {}
        if index == 2: # if objectId is new
          self.id_to_adress.append(keys[:3])
          self.id_to_targets.append(target)
      where = where[key]
    where[keys[-1]] = value 

  def get(self, keys):
    result = self._root
    for key in keys:
      result = result[key]
    return result

  def flatten(self, keys):
    root = self._root
    for key in keys:
      root = root[key]
    return _flatten(root)

a = MultiLevelDict()
a.add(['ali', 'a'], 1)
a.add(['ali', 'b'], 2)
a.add(['al', 'i'], 3)
assert a.get(['ali', 'a']) == 1
assert a.get(['ali', 'b']) == 2
assert a.get(['al', 'i']) == 3
assert a.flatten([]) == [1,2,3]