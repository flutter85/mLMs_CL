# mLMs_CL
# Introduction
**`A Multilingual Sentiment Analysis Model based on Continual Learning`**, Data Analysis and Knowledge Discovery 2023  


**`Abstract`**：`[Objective]` This study aims to address the performance degradation of multilingual 
models in handling new language tasks due to catastrophic forgetting. `[Methods]` A continual 
learning-based multilingual sentiment analysis model, mLMs-EWC, was proposed. By integrating 
the continual learning idea into the multilingual models, these models can acquire new language 
features while retaining the linguistic characteristics of previously learned languages. `[Results]`
The mLMs-EWC model outperforms the Multi-BERT model by approximately 5.2% and 4.5% on 
French and English tasks, respectively. In addition, we also evaluate our approach on a
lightweight distillation model, which showed a remarkable improvement rate of 24.7% on the 
English task. `[Limitations]` This study focuses on three widely used languages, and further 
validation is needed for the generalization ability of other languages. `[Conclusions]` The 
proposed model can alleviate catastrophic forgetting in multilingual sentiment analysis tasks and 
achieve continual learning on multilingual datasets.



# Environment
The code is based on Python 3.7.11 and PyTorch 1.11.0 version. The code is developed and tested using one NVIDIA TESLA T4.

# Data_Preprocessing
* Data should save in: `/data/${dataset}/train/*.txt`, `/data/${dataset}/valid/*.txt`, `/data/${dataset}/test/*.txt`.
* The preprocessed data are available in [data_filter](https://github.com/flutter85/mLMs_CL/tree/main/data_filter "悬停显示").

# Monolingual
Taking XLM model as an example, please use the appropriate linguistic datasets and models when training.
* `XLM_C.py`

# Bilingual
In this paper, we adopt two language training sequences: English-French-Chinese, and Chinese-French-English. Specific experimental results are detailed in the paper.  
**Note:** Under bilingual training, whether or not to add a continual learning method produced different results for the experiment, thus requiring separate training. 

* `XLM_CE.py` (Taking English-French-Chinese sequence as an example)
* `testrun2.py`
* `XLM_EWC2.py`

# Trilingual
Same as Bilingual Training above

* `XLM_CEF.py` (Taking English-French-Chinese sequence as an example)
* `testrun3.py`
* `XLM_EWC3.py`
  
