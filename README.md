# loracap: empowering unimodal ViT and LM for image captioning

- `train.py` is the file to train the model and log/visualize on the way. Simply running 

```bash
$ python train.py --dataset /path/to/your/dataset --val_dataset /path/to/val/dataset
```

suffice for default hyperparameter setups. There are HPs that you can choose from the command line, such as lr, batch size and gpu-id.
