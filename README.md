# Mytransformer
Small transformer with encoder and decoder implemented by Peng.

## Install requirements
Just run the command below to install requirements:<br>
`pip install ./Mytransformer`

## Train
The training script is in train_main.py. Change the training parameters here, changing args=ModelArgs(...) to chang model parameters

## Predict
The training script is in predict_main.py. The model and vocabulary parameters need to be the same during prediction as during training.

## Remark
This version of the vocabulary treats each word as a token. The vocabulary size is about 200, while that of llama3's tokenizer is 128,000. This time, the results are as good as those in "dive into deep learning", which means that my model code is correct.<br>
I did not implement kv-cache. kv-cache is a technic that speeds up inference and is not used during training. The implementation in "Dive into deep learning" is wrong, which doesn't reduce any computation.

## References
https://github.com/meta-llama/llama3<br>
https://zh-v2.d2l.ai/