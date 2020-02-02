import math, random
from os import listdir

################################################################################
# Part 0: Utility Functions
################################################################################

COUNTRY_CODES = ['af', 'cn', 'de', 'fi', 'fr', 'in', 'ir', 'pk', 'za']

def start_pad(n):
    ''' Returns a padding string of length n to append to the front of text
        as a pre-processing step to building n-grams '''
    return '~' * n

def ngrams(n, text):
    ''' Returns the ngrams of the text as tuples where the first element is
        the length-n context and the second is the character '''

    to_return = []
    padded_text = start_pad(n) + text
    i = n
    while i < len(padded_text):
        to_return.append((padded_text[(i - n):i], text[i - n]))
        i = i + 1
    return to_return

def create_ngram_model(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        model.update(f.read())
    return model

def create_ngram_model_lines(model_class, path, n=2, k=0):
    ''' Creates and returns a new n-gram model trained on the city names
        found in the path file '''
    model = model_class(n, k)
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            model.update(line.strip())
    return model

################################################################################
# Part 1: Basic N-Gram Model
################################################################################

class NgramModel(object):
    ''' A basic n-gram model using add-k smoothing '''

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.vocab_count = dict()
        self.context_count = dict()
        self.ngrams = dict()
        pass

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.vocab_count.keys())

    def update(self, text):
        ''' Update the context dictionary '''
        curr_ngram = ngrams(self.n, text)
        for ngram in curr_ngram:
            if ngram not in self.ngrams:
                self.ngrams[ngram] = 1
            else:
                self.ngrams[ngram] += 1
            if ngram[0] not in self.context_count:
                self.context_count[ngram[0]] = 1
            else:
                self.context_count[ngram[0]] += 1
            
            if ngram[1] not in self.vocab_count:
                self.vocab_count[ngram[1]] = 1
            else:
                self.vocab_count[ngram[1]] += 1

    def prob(self, context, char):
        ''' Returns the probability of char appearing after context '''
        ''' ngram(self.text, vocab.get(context+char) +k / count(all context) +k*V)'''
        ''' stuff = ngram(self.n, self.text) '''

        V = len(self.vocab_count)
        if context not in self.context_count:
            return 1 / V
        
        # Check for (context + char)
        count = 0 if ngram not in self.ngrams else self.ngrams[ngram]

        return (count + self.k) / (self.context_count[context] + self.k * V)

    def random_char(self, context):
        ''' Returns a random character based on the given context and the 
            n-grams learned by this model '''
        r = random.random()
        sortedVocab = sorted(list(self.vocab_count.keys()))
        for i in range(len(sortedVocab)):
            probSum = 0
            for j in range(i):
                probSum += self.prob(context, sortedVocab[j])

            if probSum <= r and r < probSum + self.prob(context, sortedVocab[i]):
                return sortedVocab[i]


        return sortedVocab[i]


    def random_text(self, length):  
        ''' Returns text of the specified character length based on the
            n-grams learned by this model '''
        text = ''
        context = ''
        for i in range(self.n):
            context += '~'
        
        for i in range(length):
            # Generate a random character
            char = self.random_char(context)
            text += char

            # Update context by replacing last character in context with generated char
            # and removing the first character
            context += char
            context = context[1:]

        return text
        

    def perplexity(self, text):
        ''' Returns the perplexity of text based on the n-grams learned by
            this model '''

        text_ngram = ngrams(self.n, text)
        sum_logs = 0

        for i in text_ngram:
            probability = self.prob(i[0], i[1])
            if probability == 0:
                return float('inf')
            sum_logs += math.log(1/ probability)

        sum_logs = math.exp(sum_logs)

        return math.pow(sum_logs, 1 / len(text_ngram)) 

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        super().__init__(n, k)
        self.lambdas = []
        for i in range(self.n + 1):
            self.lambdas.append(1/(self.n + 1))

    # def get_vocab(self):
    #     pass

    def update(self, text):
        # Generate update for n = 0 ... n
        for i in range(self.n + 1):
            curr_ngram = ngrams(i, text)
            for ngram in curr_ngram:
                if ngram not in self.ngrams:
                    self.ngrams[ngram] = 1
                else:
                    self.ngrams[ngram] += 1
                if ngram[0] not in self.context_count:
                    self.context_count[ngram[0]] = 1
                else:
                    self.context_count[ngram[0]] += 1
                
                if ngram[1] not in self.vocab_count:
                    self.vocab_count[ngram[1]] = 1
                else:
                    self.vocab_count[ngram[1]] += 1
        
        

    def prob(self, context, char):
        to_return = 0
        for i in range(0, len(self.lambdas)):
            to_return += self.lambdas[i] * super().prob(context[i:], char)

        return to_return

    '''helper function to override lambdas'''   
    def override_lambdas(self, lst):
        sum_lam = 0
        for weight in lst:
            sum_lam += weight
        if sum_lam == 1 and len(lst) == self.n + 1:
            self.lambdas = lst

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    '''Possible ideas to consider:
    --utilizing a special end-of-text character
    --trying a new method for determining the vocab
    --improving how your model handles novel characters'''

    # lookup = {0:'af', 1:'cn', 2:'de', 3:'fi', 4:'fr',
    #           5:'in', 6:'ir', 7:'pk', 8:'za'}

    # Load training data
    directory = "train/" 
    models_lst = []
    files = listdir(directory)
    
    for i in range(0, len(files)):
        models_lst.append(create_ngram_model_lines(NgramModel, directory + files[i]))

    correct = 0
    incorrect = 0
    for i in range(len(files)):
        lines = []
        with open(directory + files[i], encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                lines.append(line)
        for line in lines:
            min_perp = 100
            best_idx = 0
            for j in range(len(models_lst)):
                model = models_lst[j]
                curr_perp = model.perplexity(line)
                if curr_perp < min_perp:
                    min_perp = curr_perp
                    best_idx = j
            if best_idx == i:
                correct += 1
            else:
                incorrect += 1
        print('Percent Correct = ' + str(correct/(correct + incorrect)))
        correct = 0
        incorrect = 0


