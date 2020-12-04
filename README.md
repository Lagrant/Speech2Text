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
