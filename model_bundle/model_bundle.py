# -*- coding: utf-8 -*-

"""Main module."""

class model_bundle(object):
  model_list = [(PCA, {}), 
                # (KernelPCA, {}),
                (SparsePCA, {})]
  def __init__(self, *args, **kwargs):
    self.models = [_(**params) for _, params in self.model_list]
    self.names = [_.__class__.__name__ for _ in self.models]
    pass
  
  def set_models(self, model_list=None, **kwargs):
    """
    kwargs: {model_name: {param: value, ...}, ...}
    """
    model_dict = {}
    for model, params in model_list:
      name = model().__class__.__name__
      params.update(**kwargs.get(name, {}))
      # models.append(model(params))
      model_dict.update({name: model(params)})
    self.models = model_dict.values()
    self.names = model_dict.keys()
    return model_dict  # {name: model for name, model in models}

  def fit(self, X, y=None):
    for _ in self.models:
      _.fit(X, y)
    
  def predict(self, X):
    if all(map(hasattr, self.models, 'predict')):
      df = pd.DataFrame({_.__class__.__name__: _.predict(X) for _ in self.models})
    else:
      raise 'not all models have .predict method'
    return df

  def transform(self, X):
    if all(map(hasattr, self.models, 'transform')):
      return {_.__class__.__name__: _.transform(X) for _ in self.models}
    else:
      raise 'not all models have .transform method'

models = model_bundle()
models.fit(X)
models.transform(X)