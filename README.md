# Mytransformer

Just run the command below to install requirements:
pip install ./Mytransformer


我尝试复现的transformer

At least now the code works and the loss goes down during training. However, the model translation accuracy is almost equal to none. I don't know why, maybe it's the tokenizer. Maybe the vocabulary size is too big...

Change the training hyperparameters in train_main.py. Change the model architecture parameters in class ModelAars in model.py.