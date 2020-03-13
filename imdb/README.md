# IMDB Experiment

You need to:

- Download the imdb cropped faces dataset from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar and put it into the /data folder
- Download the resnet_v1_50.ckpt checkpoint for ResNet-50 pretrained on ImageNet and put it into the /data folder
- Extract auxiliary.zip files into the /data folder  

To run the code execute the imdb_main.py file. Have a look at the imdb_parser.py file to see the command line parameters.


Example of usage:

```
# run baseline model
python3 imdb_main.py --exp_name'eb1' --lmb 0

# run our model
python3 imdb_main.py --exp_name'eb1' --lmb 0.9
```


you can try different values for lmb. The higher lmb, the more the representation is unbiased. Be careful that
for values of lmb too high, the model can not fit the training data.
