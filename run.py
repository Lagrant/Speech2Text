from transcriber import main as tscb
from translator import main as tslt
from transcriber import transcribe
from translator import translate

def eval(path):
    sentence = transcribe(path, model)
    result = translate(sentence, transformer, tokenizer_tgt, tokenizer_src, MAX_LENGTH)
    return result

model = tscb()
transformer, tokenizer_tgt, tokenizer_src, MAX_LENGTH = tslt()

eval('./results/audio/test1.wav')