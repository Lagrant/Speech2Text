# Speech2Text

Acoustic encoder -> Shrink mechanism -> Sementic encoder -> Decoder

## Acoustic encoder

Speech input -> Pre-net -> Multiple stacked self-attention layers -> Softmax -> CTC loss function

Pre-net is a linear model that reshapes speech input matrix to model hidden size, it extracts acoustic features.
CTC is used to define the loss function
