LongBuilder:  
This repo is the builder of Longformer model from one RoBERTa checkpoint in Chinese.  

How to use it:  
1. Use `data_preprocess.py` to preprocess the raw data
``` 
python3 data_preprocess.py --dataset='news'
```
2. Use `builder.py` to modify the roberta_zh model in `./roberta_checkpoint` and pretraining
``` 
python3 builder.py --dataset='news' --order='0'
```  
Script Flow:
1. Evaluate the RoBERTa checkpoint on the created val set with Masked Language Modeling task.
2. Modify the given model into Longformer model.
3. Pretrain the model.
4. Evaluate on MLM task.

Feature:  
* Our model is based on Roberta_zh model from https://github.com/brightmart/roberta_zh  
* The script is based on https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb  
* Different with original script, to fit Chinese Language, we use Whole Word Masking tech in the pretraining. `data_Collator.py`
and `Longdataset.py` is rewrote to implement WWM.
* The pretrained model is in `./save_model`