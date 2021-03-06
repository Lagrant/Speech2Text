# Speech2Text

Acoustic encoder -> Acoustic decoder -> Sementic encoder -> Decoder

## Acoustic encoder

Speech input -> Pre-net -> Transformer -> Label smoothing loss function

Pre-net is a module that has two 2D-convolutional layers followed by two stacked attention layers.
Transformer is the main structure of the encoder, and [here](http://jalammar.github.io/illustrated-transformer/) is an article to illustrate it. 
Label smoothing is the loss function for classification problems to prevent the model from predicting the training examples too confidently.

## Acoustic decoder
Outputs -> Embeddings -> Positional encoding -> Transformer -> Softmax -> Output probabilities

Positional encoding provides relative positional information for a word at the sentence such that words sharing similar meaning and position can be closer in d-dimensional space.

## Semantic encoder and decoder

They have similar structure to acoustic encoder and decoder. The loss function of semantic encoder is sparse categorical cross entropy loss.

---
We can look at the model at two perspectives:
1. Acoustic encoder, acoustic decoder and semantic encoder can be regarded as an integrated encoder, it extracts information from source speech. Decoder transforms information to target language.

2. Acoustic encoder and acoustic decoder can be regarded as a transcriber that transcribes speech to text. Semantic encoder and decoder can be thought as a NMT model that translates the text to target language.

## Sources
A pytorch version of speech transformer model: https://github.com/kaituoxu/Speech-Transformer

A bash tutoral to implement parallelization of batch jobs: https://jerkwin.github.io/2013/12/14/Bash%E8%84%9A%E6%9C%AC%E5%AE%9E%E7%8E%B0%E6%89%B9%E9%87%8F%E4%BD%9C%E4%B8%9A%E5%B9%B6%E8%A1%8C%E5%8C%96/

Coverting flac file to wav file: http://magento4newbies.blogspot.com/2014/11/converting-wav-files-to-flac-with.html

## Modifications
We reimplemented almost all the classes and functions by tensorflow at ./model/modules/ . The original source is at https://github.com/kaituoxu/Speech-Transformer/src/transformer

## How to run
Specify path of speech corpus at ./config/hparams_transcriber.yaml
Specify path of text corpus at ./config/hparams_translator.yaml
You can also set other hyper-parameters at ./config/hparams_transcriber.yaml and ./config/hparams_translator.yaml

Then, you just need
> python run.py

Or
> python transcriber.py

> python translator.py

To evaluate results, you need to build a directory named results at the root directory, store results at results/candidate.txt and ground truth at results/ref.txt. Then run
> python evaluation.py

## Major softwares
python=3.6.1

nltk==3.5

tensorflow==2.3.1

tensorflow-datasets==4.1.0

speechpy==2.4

## Datasets
1. [Open Translation Project](https://www.ted.com/participate/translate) from Ted is a set of subtitles in many languages.

2. [LibriSpeech](http://www.openslr.org/12/) is an English speech corpus.

## References
Rafael Müller, Simon Kornblith, Geoffrey Hinton, "When Does Label Smoothing Help?" (arXiv:1906.02629 [cs.LG]).

Yuchen Liu, Junnan Zhu, Jiajun Zhang, and Chengqing Zong, "Bridging the Modality Gap for Speech-to-Text Translation" (arXiv:2010.14920 [cs.CL]).

C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 2818-2826, doi: 10.1109/CVPR.2016.308.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (p./pp. 5998--6008), .

L. Dong, S. Xu and B. Xu, "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition," 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Calgary, AB, 2018, pp. 5884-5888, doi: 10.1109/ICASSP.2018.8462506.


