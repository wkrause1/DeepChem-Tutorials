import deepchem as dc
import numpy as np

tasks, datasets, transformers = dc.molnet.load_delaney(featurizer="GraphConv")
train_dataset, valid_dataset, test_dataset = datasets

for X, y, w, id in test_dataset.itersamples():
    print(y, id)

for X, y, w, ids in test_dataset.iterbatches(batch_size=50):
    print(y.shape)

test_dataset.to_dataframe()
for X, y, w, ids in test_dataset.itersamples():
    print(X, y, w, ids)

X = np.random.random((10,5))
y = np.random.random((10,2))
dataset = dc.data.NumpyDataset(X=X, y=y)
print(dataset)