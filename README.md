# Speech2Text

Acoustic encoder -> Shrink mechanism -> Sementic encoder -> Decoder

## Acoustic encoder

Speech input -> Pre-net -> Transformer -> CTC loss function

Pre-net is a linear model that reshapes speech input matrix to model hidden size, it extracts acoustic features.
Transformer is the main structure of the encoder. [Here](http://jalammar.github.io/illustrated-transformer/) is an article to illustrate it.
CTC is the loss function which is widely used in speech recognition. [Here](https://distill.pub/2017/ctc/) is an introduction to CTC.

## Shrink mechanism
Shrink mechanism is a mask that hides blank label or consecutively repeated label generated in CTC. This is used to reshape states to fit semantic encoder.

## Semantic encoder and decoder

It adapts NMT model. The loss function of semantic encoder is AD loss, the loss function of decoder is cross entropy. Decoder also adopts transformer as the basic model structure.

---
We can look at the model at two perspectives:
1. Acoustic encoder, shrink mechanism and semantic encoder can be regarded as an integrated encoder, it extracts information from source speech. Decoder transforms information to target language.

2. Acoustic encoder and shrink mechanism can be regarded as a transcriber that transcribes speech to text. Semantic encoder and decoder can be thought as a NMT model that translates the text to target language.

## Datasets
1. [Open Translation Project](https://www.ted.com/participate/translate) from Ted is a set of subtitles in many languages.

2. [LibrariSpeech](http://www.openslr.org/12/) is an English speech corpus.

## Reference
A pytorch version of speech transformer model. Click [here](https://github.com/kaituoxu/Speech-Transformer) to see.

A tensorflow version of transformer model for language understanding. Click [here](https://www.tensorflow.org/tutorials/text/transformer) to see.

A lib to convert flac file to wav file. Click [here](http://magento4newbies.blogspot.com/2014/11/converting-wav-files-to-flac-with.html) to see.

Yuchen Liu, Junnan Zhu, Jiajun Zhang, and Chengqing Zong, "Bridging the Modality Gap for Speech-to-Text Translation".

C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 2818-2826, doi: 10.1109/CVPR.2016.308.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (p./pp. 5998--6008), .

Linhao Dong, Shuang Xu,and Bo Xu. “Speech-transformer:A no-recurrence sequence-to-sequence model for speech recognition” in ICASSP 2018



