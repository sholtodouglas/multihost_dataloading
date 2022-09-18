# multihost_dataloading
Experimenting with how best to do multi-host dataloading. Once we determine a reccomended option, we will likely upstream it to jax's GDA lib as a function which takes a tf.data.Dataset and a GDA definition, and returns an efficient iterator that returns the desired GDA.


Install with the following

```
git clone https://github.com/sholtodouglas/multihost_dataloading
cd multihost_dataloading
pip install -e .

python3 multihost_dataloading/tests.py
```

# Methods tested

## All data loaded by all hosts (strawman)


## Per replica data pipeline

## Per host data pipeline

## Data is loaded fully sharded across all devices, and resharded inside pjit (Pax method)
