# Mytransformer
Small transformer with encoder and decoder implemented by Peng.<br>
This is just a runnable demo.

## Install requirements
Run the command below to install requirements:<br>
`pip install ./Mytransformer`<br>
You may need python<=3.10 to successfully install blobfile. My python==3.9.13 .<br>
You also need to download _tokenizer.model_ in [llama3](https://huggingface.co/meta-llama), which needs a HuggingFace account and request. Or you can switch to branch "vocab_version" without download a tokenizer.  


## Train
The training script is in train_main.py. Change the hyperparameters here, changing args=ModelArgs(...) to chang model parameters.

## Predict
The training script is in predict_main.py. The model and vocabulary parameters need to be the same during prediction as during training

## Remark
At least now the code works and the loss goes down during training. However, the model translation accuracy is almost equal to none. I don't know why, maybe it's the tokenizer. Maybe the vocabulary size is too big...<br>
New: The poor performance of the model is caused by a large vocabulary (the vocabulary dimension is 128000). Consider increasing the model size and increasing the amount of training data to compensate.

## References
https://github.com/meta-llama/llama3<br>
https://zh-v2.d2l.ai/