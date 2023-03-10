

# Setup

The following command will install the packages according to the configuration file ```requirements.txt```:

```
$ pip install -r requirements.txt
```

# Model Training and Testing

The following command will train the model:

```
$ python train.py
```
Currently, two devices (i.e. "lightning.qubit" and "default.qubit") can be specified in the file ```train.py```. Best models will be saved in the folder ```checkpoints``` after the training. The folder ```data``` contains the raw data. The folder ```plot_data``` contains the prediction and target data for model performance visualization.

The following command will evaluate the model:

```
$ python evaluation.py
```
We have already placed the saved model files in the folder ```checkpoints```. You can directly run the above code for evaluation if you do not want to spend time in training the models.
