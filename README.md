# Image colorization

To colorize image run:

```python model_predict.py <SOURCE IMAGE> <OUTPUT IMAGE NAME>```

For example:

```python model_predict.py test_imgs/1.jpg images/test.png```

Arguments:

```
positional arguments:
  img                   Black-white image file
  output                Output directory and file name

optional arguments:
  -h, --help            show this help message and exit
  -c CHECKPOINT, --checkpoint CHECKPOINT
                        Directory of the model checkpoint to use. Defaults to already pretrained one
                        ```