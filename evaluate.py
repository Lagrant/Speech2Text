import nltk

def read_file(path):
    lst = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            lst.append(line.split(' '))
            line = f.readline()
    return lst

if __name__ == "__main__":
    ref = read_file('./results/ref.txt')
    cand = read_file('./results/candidate.txt')
    scores = []
    for r, c in zip(ref, cand):
        score = nltk.translate.bleu_score.sentence_bleu([r], c, weights = (0.5, 0.5))
        scores.append(score)
    print(scores)