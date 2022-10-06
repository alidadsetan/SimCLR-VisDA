class MultiLevelDict(object):
  def __init__(self):
    self._root = {}

  def add(self, keys, value):
    where = self._root
    for key in keys[:-1]:
      if not key in where:
        where[key] = {}
      where = where[key]
    where[keys[-1]] = value 

  def get(self, keys):
    result = self._root
    for key in keys:
      result = result[key]
    return result

a = MultiLevelDict()
a.add(['ali', 'a'], 1)
a.add(['ali', 'b'], 2)
a.add(['al', 'i'], 3)
assert a.get(['ali', 'a']) == 1
assert a.get(['ali', 'b']) == 2
assert a.get(['al', 'i']) == 3