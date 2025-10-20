import unicodedata, re
from pathlib import Path
from collections import Counter
import math
import random

def normalize_sentence(s, replace_numbers = False):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")
    s = s.replace("—", "-").replace("–", "-").replace("…", "...")
    s = s.lower()
    if replace_numbers:
        s = re.sub(r"\d+([.,]\d+)*", "<num>", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
def load_and_normalize(filepath: Path, replace_numbers=False):
    with filepath.open("r", encoding="utf-8", errors="replace") as f:
        lines = [normalize_sentence(line, replace_numbers) for line in f if line.strip()]
    lines = [f"<s> {line} </s>" for line in lines if line]
    return lines
def build_vocab(sentences, min_freq=2):
    counter = Counter()
    for line in sentences:
        for token in line.split():
            counter[token] += 1
    vocab = {tok for tok, c in counter.items() if c >= min_freq}
    vocab.add("<UNK>")
    return vocab, counter
def replace_rare(sentences, vocab):
    new_sentences = []
    for line in sentences:
        tokens = line.split()
        new_line = " ".join(tok if tok in vocab else "<UNK>" for tok in tokens)
        new_sentences.append(new_line)
    return new_sentences
def save_sentences(sentences, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for line in sentences:
            f.write(line + "\n")
base_raw = Path('/Users/nsumesh/Documents/GitHub/642HW2/ptbdataset')
base_proc = Path('/Users/nsumesh/Documents/GitHub/642HW2/normalizedptbdataset')
files = {
    "train": "ptb.train.txt",
    "valid": "ptb.valid.txt",
    "test":  "ptb.test.txt"
}
train_sentences = load_and_normalize(base_raw / files["train"], replace_numbers=True)
valid_sentences = load_and_normalize(base_raw / files["valid"], replace_numbers=True)
test_sentences  = load_and_normalize(base_raw / files["test"],  replace_numbers=True)
vocab, counter = build_vocab(train_sentences, min_freq=2)
train_sentences = replace_rare(train_sentences, vocab)
valid_sentences = replace_rare(valid_sentences, vocab)
test_sentences  = replace_rare(test_sentences, vocab)
save_sentences(train_sentences, base_proc / "train.final.txt")
save_sentences(valid_sentences, base_proc / "valid.final.txt")
save_sentences(test_sentences,  base_proc / "test.final.txt")
with (base_proc / "vocab.txt").open("w", encoding="utf-8") as vf:
    for token in sorted(vocab):
        vf.write(token + "\n")

def read_sentences(path):
    sents = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sents.append(line.split())
    return sents
def build_n_gram_counts(sentences, n):
    ngram_counts = Counter() 
    context_counts = Counter()
    for sentence in sentences:
        length = len(sentence)
        if length<n:
            continue
        for i in range(length-n+1):
            ngram = tuple(sentence[i:i+n])
            ngram_counts[ngram]+=1
            if n>1:
                context = ngram[:-1]
                context_counts[context]+=1
    return ngram_counts, context_counts

def building_ngrams(sentences, max_n = 4):
    results = {}
    for i in range(1, max_n+1):
        results[i] = build_n_gram_counts(sentences,i)
    return results

path_dir_training = Path('/Users/nsumesh/Documents/GitHub/642HW2/normalizedptbdataset/train.final.txt')
path_dir_validation = Path('/Users/nsumesh/Documents/GitHub/642HW2/normalizedptbdataset/valid.final.txt')
path_dir_test = Path('/Users/nsumesh/Documents/GitHub/642HW2/normalizedptbdataset/test.final.txt')
training_sentences = read_sentences(path_dir_training)
validation_sentences = read_sentences(path_dir_validation)
testing_sentences = read_sentences(path_dir_test)
all_counts = building_ngrams(training_sentences, max_n=4)
unigram_count, unigram_context = all_counts[1]
bigram_count, bigram_context = all_counts[2]
trigram_count, trigram_context = all_counts[3]
fourgram_count, fourgram_context = all_counts[4]

def mle_probabilities(word, context, ngram_count, context_count):
    ngram = context + (word,)
    frequency = ngram_count.get(ngram,0)
    if(len(context)==0):
        denominator = sum(ngram_count.values())
    else:
        denominator = context_count.get(context,0)
    if denominator==0.0 or frequency==0.0:
        return 0.0
    return frequency/denominator

def compute_perplexity(sentences, ngram_count, context_count, n):
    total_log_probabilities = 0.0
    tokens = 0
    for sentence in sentences:
        for i in range(n-1, len(sentence)):
            context = tuple(sentence[i-n+1:i]) if n>1 else ()
            word = sentence[i]
            mle_probability = mle_probabilities(word, context, ngram_count, context_count)
            if mle_probability==0.0:
                return float("inf")
            total_log_probabilities+=math.log2(mle_probability)
            tokens+=1
    if tokens==0.0:
        return float("inf")
    average_log_probabilities = total_log_probabilities/tokens
    return 2**(-average_log_probabilities)

unigram_perplexity = compute_perplexity(training_sentences, unigram_count, unigram_context, n=1)
bigram_perplexity = compute_perplexity(training_sentences, bigram_count, bigram_context, n=2)
trigram_perplexity = compute_perplexity(training_sentences, trigram_count, trigram_context, n=3)
fourgram_perplexity = compute_perplexity(training_sentences, fourgram_count, fourgram_context, n=4)

print("Unigram Perplexity: ", unigram_perplexity)
print("Bigram Perplexity: ", bigram_perplexity)
print("Trigram Perplexity: ", trigram_perplexity)
print("Four gram Perplexity: ", fourgram_perplexity)
print()
def laplace_probability(word, context, ngram_count, context_count):
    ngram = context + (word,)
    frequency = ngram_count.get(ngram,0)+1
    if(len(context)==0):
        denominator = sum(ngram_count.values()) + len(vocab)
    else:
        denominator = context_count.get(context,0) + len(vocab)
    return frequency/denominator 

def compute_perplexity_with_laplace(sentences, ngram_count, context_count, n):
    total_log_probabilities = 0.0
    tokens = 0
    for sentence in sentences:
        for i in range(n-1, len(sentence)):
            context = tuple(sentence[i-n+1:i]) if n>1 else ()
            word = sentence[i]
            laplace_prob = laplace_probability(word, context, ngram_count, context_count)
            if laplace_prob==0.0:
                return float("inf")
            total_log_probabilities+=math.log2(laplace_prob)
            tokens+=1
    if tokens==0.0:
        return float("inf")
    average_log_probabilities = total_log_probabilities/tokens
    return 2**(-average_log_probabilities)

unigram_perplexity_with_laplace = compute_perplexity_with_laplace(training_sentences, unigram_count, unigram_context, n=1)
bigram_perplexity_with_laplace = compute_perplexity_with_laplace(training_sentences, bigram_count, bigram_context, n=2)
trigram_perplexity_with_laplace = compute_perplexity_with_laplace(training_sentences, trigram_count, trigram_context, n=3)
fourgram_perplexity_with_laplace = compute_perplexity_with_laplace(training_sentences, fourgram_count, fourgram_context, n=4)

print("Unigram Perplexity with Laplace: ", unigram_perplexity_with_laplace)
print("Bigram Perplexity with Laplace: ", bigram_perplexity_with_laplace)
print("Trigram Perplexity with Laplace: ", trigram_perplexity_with_laplace)
print("Fourgram Perplexity with Laplace: ", fourgram_perplexity_with_laplace)
print()

def interpolation(word, context, unigram_count, bigram_count, bigram_context, trigram_count, trigram_context, fourgram_count, fourgram_context, lambdas, vocab_size):
    lambda1, lambda2, lambda3, lambda4 = lambdas[0], lambdas[1], lambdas[2], lambdas[3]
    unigram_probability = (unigram_count.get((word,), 0) + 1) / (sum(unigram_count.values()) + vocab_size)
    if len(context) >= 1:
        context_bi = (context[-1],)
        bigram_probability = (bigram_count.get(context_bi + (word,), 0) + 1) / (bigram_context.get(context_bi, 0) + vocab_size)
    else:
        bigram_probability = unigram_probability
    if len(context) >= 2:
        context_tri = tuple(context[-2:])
        trigram_probability = (trigram_count.get(context_tri + (word,), 0) + 1) / (trigram_context.get(context_tri, 0) + vocab_size)
    else:
        trigram_probability = bigram_probability
    if len(context) >= 3:
        context_four = tuple(context[-3:])
        fourgram_probability = (fourgram_count.get(context_four + (word,), 0) + 1) / (fourgram_context.get(context_four, 0) + vocab_size)
    else:
        fourgram_probability = trigram_probability
    return lambda1 * unigram_probability + lambda2 * bigram_probability + lambda3 * trigram_probability + lambda4 * fourgram_probability


def interpolation_perplexity(sentences, unigram_count, bigram_count, bigram_context, trigram_count, trigram_context, fourgram_count, fourgram_context, lambdas, vocab_size):
    total_log_probabilities = 0.0
    tokens = 0
    for sentence in sentences:
        for i in range(3, len(sentence)):  
            context = tuple(sentence[i-3:i])
            probability = interpolation(sentence[i], context,unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,lambdas, vocab_size)
            if probability <= 0:
                continue
            total_log_probabilities += math.log2(probability)
            tokens += 1
    average_log_probabilities = total_log_probabilities / tokens
    return 2 ** (-average_log_probabilities)

lambda_weights = [
    [0.10, 0.15, 0.35, 0.40],
    [0.05, 0.05, 0.30, 0.60],
    [0.25, 0.10, 0.50, 0.15],
    [0.40, 0.20, 0.20, 0.20],
    [0.01, 0.24, 0.25, 0.50],
    [0.33, 0.17, 0.25, 0.25],
    [0.12, 0.18, 0.27, 0.43],
    [0.07, 0.28, 0.10, 0.55],
    [0.22, 0.22, 0.22, 0.34],
    [0.48, 0.12, 0.30, 0.10],
    [0.26, 0.14, 0.31, 0.29],
    [0.09, 0.41, 0.19, 0.31]
]

vocab_size = len(vocab)
highest_perplexity = float("inf")
best_lambda = None

for lambda_w in lambda_weights:
    interpolation_perplexity_prob = interpolation_perplexity(validation_sentences, unigram_count, bigram_count, bigram_context, trigram_count, trigram_context, fourgram_count, fourgram_context, lambda_w, vocab_size)
    if interpolation_perplexity_prob < highest_perplexity:
        highest_perplexity, best_lambda = interpolation_perplexity_prob, lambda_w

print("Best Lambda:", best_lambda)
print("Lowest Perplexity:", highest_perplexity)
print()
def backoff_probability(word, context, unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context, alpha=0.4):
    if len(context) >= 3:
        context_four = tuple(context[-3:])
        fourgram_freq = fourgram_count.get(context_four + (word,), 0)
        if fourgram_freq > 0:
            return fourgram_freq / fourgram_context.get(context_four, 1)
        return alpha * backoff_probability(word, context[-2:],unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context, alpha=alpha)

    if len(context) >= 2:
        context_tri = tuple(context[-2:])
        trigram_freq = trigram_count.get(context_tri + (word,), 0)
        if trigram_freq > 0:
            return trigram_freq / trigram_context.get(context_tri, 1)
        return alpha * backoff_probability(word, context[-1:], unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context, alpha=alpha)

    if len(context) >= 1:
        context_bi = (context[-1],)
        bigram_freq = bigram_count.get(context_bi + (word,), 0)
        if bigram_freq > 0:
            return bigram_freq / bigram_context.get(context_bi, 1)
        return alpha * backoff_probability(word, (),unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context, alpha=alpha)

    total = sum(unigram_count.values())
    return unigram_count.get((word,), 0) / total


def backoff_perplexity(sentences,unigram_count, bigram_count, bigram_context, trigram_count, trigram_context, fourgram_count, fourgram_context, alpha=0.4):
    total_log_probabilities, tokens = 0.0, 0
    for sentence in sentences:
        for i in range(3, len(sentence)):
            context = tuple(sentence[i - 3:i])
            probability = backoff_probability(sentence[i], context,unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha)
            if probability == 0.0:
                continue
            total_log_probabilities += math.log2(probability)
            tokens += 1
    if tokens == 0:
        return float("inf")
    avg_log_probability = total_log_probabilities / tokens
    return 2 ** (-avg_log_probability)

alpha = 0.4
boff_perplexity = backoff_perplexity(validation_sentences,unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha)
print("Backoff Perplexity:", boff_perplexity)
print()

mle_perplexity = compute_perplexity(test_sentences, unigram_count, unigram_context, n=1)
test_unigram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, unigram_count, unigram_context, n=1)
test_bigram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, bigram_count, bigram_context, n=2)
test_trigram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, trigram_count, trigram_context, n=3)
test_fourgram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, fourgram_count, fourgram_context, n=4)
lambdas = [0.4, 0.2, 0.2, 0.2]
test_interpolation_perplexity = interpolation_perplexity(testing_sentences, unigram_count, bigram_count, bigram_context, trigram_count, trigram_context, fourgram_count, fourgram_context, lambdas, vocab_size)
test_backoff_perplexity = backoff_perplexity(testing_sentences, unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha=0.4)

print('MLE Perplexity on Testing Set: ',mle_perplexity)
print('Unigram Laplace Perplexity on Testing Set: ',test_unigram_laplace_perplexity)
print('Bigram Laplace Perplexity on Testing Set: ',test_bigram_laplace_perplexity)
print('Trigram Laplace Perplexity on Testing Set: ',test_trigram_laplace_perplexity)
print('Fourgram Laplace Perplexity on Testing Set: ',test_fourgram_laplace_perplexity)
print('Interpolation perplexity on Testing Set: ',test_interpolation_perplexity)
print('Backoff Perplexity for Testing set: ', test_backoff_perplexity)
print()

mle_perplexity_unigram = compute_perplexity(testing_sentences, unigram_count, unigram_context, n=1)
mle_perplexity_bigram = compute_perplexity(testing_sentences, bigram_count, bigram_context, n=2)
mle_perplexity_trigram = compute_perplexity(testing_sentences, trigram_count, trigram_context, n=3)
mle_perplexity_fourgram = compute_perplexity(testing_sentences, fourgram_count, fourgram_context, n=4)


test_unigram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, unigram_count, unigram_context, n=1)
test_bigram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, bigram_count, bigram_context, n=2)
test_trigram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, trigram_count, trigram_context, n=3)
test_fourgram_laplace_perplexity = compute_perplexity_with_laplace(testing_sentences, fourgram_count, fourgram_context, n=4)
lambdas = [0.4, 0.2, 0.2, 0.2]
test_interpolation_perplexity = interpolation_perplexity(testing_sentences, unigram_count, bigram_count, bigram_context, trigram_count, trigram_context, fourgram_count, fourgram_context, lambdas, vocab_size)
test_backoff_perplexity = backoff_perplexity(testing_sentences, unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha=0.4)

print('MLE Perplexity on Testing Set for Unigram: ',mle_perplexity_unigram)
print('MLE Perplexity on Testing Set for Bigram: ',mle_perplexity_bigram)
print('MLE Perplexity on Testing Set for Trigram: ',mle_perplexity_trigram)
print('MLE Perplexity on Testing Set for Fourgram: ',mle_perplexity_fourgram)

print('Unigram Laplace Perplexity on Testing Set: ',test_unigram_laplace_perplexity)
print('Bigram Laplace Perplexity on Testing Set: ',test_bigram_laplace_perplexity)
print('Trigram Laplace Perplexity on Testing Set: ',test_trigram_laplace_perplexity)
print('Fourgram Laplace Perplexity on Testing Set: ',test_fourgram_laplace_perplexity)
print('Interpolation perplexity on Testing Set: ',test_interpolation_perplexity)
print('Backoff Perplexity for Testing set: ', test_backoff_perplexity)


def generate_sentences(unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha=0.4):
    start_token = [random.choice(list(unigram_count.keys()))[0]]
    sentence = list(start_token)
    for _ in range(15-len(start_token)):
        context = tuple(sentence[-3:])
        candidates = []
        probs = []
        for word in vocab:
            if word in ['<s>','</s>','<UNK>', '<unk>']:
                continue
            prob = backoff_probability(word, context, unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha=0.4)
            if prob>0:
                candidates.append(word)
                probs.append(prob)
        if not candidates:
            break
        word = random.choices(candidates, weights=probs, k=1)[0]
        sentence.append(word)
        if word in ['.','!','?']:
            break
    return " ".join(sentence)

for i in range(5):
    sentence = generate_sentences(unigram_count, bigram_count, bigram_context,trigram_count, trigram_context,fourgram_count, fourgram_context,alpha=0.4)
    print("Sentence Generated: ",sentence)