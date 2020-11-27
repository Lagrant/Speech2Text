# Speech2Text

Acoustic encoder -> Shrink mechanism -> Sementic encoder -> Decoder

## Acoustic encoder

Speech input -> Pre-net -> Multiple stacked self-attention layers -> Feedforwad layer -> Softmax -> CTC loss function

Pre-net is a linear model that reshapes speech input matrix to model hidden size, it extracts acoustic features.
CTC is used to define the loss function

## Shrink mechanism
Shrink mechanism is a mask that hides blank label or consecutively repeated label generated in CTC. This is used to reshape states to fit semantic encoder.

## Semantic encoder and decoder

It adapts NMT model
