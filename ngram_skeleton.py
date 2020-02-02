import math, random

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
        to_return.append([padded_text[(i - n):i], text[i - n]])
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
        self.ngrams = []
        pass

    def get_vocab(self):
        ''' Returns the set of characters in the vocab '''
        return set(self.vocab_count.keys())

    def update(self, text):
        ''' Update the context dictionary '''
        curr_ngram = ngrams(self.n, text)
        self.ngrams += curr_ngram

        for ngram in curr_ngram:
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
        count = 0
        for ngram in self.ngrams:
            if ngram == [context, char]:
                count += 1

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

        # for i in text_ngram:
        #     sum_logs *= self.prob(i[0], i[1])

        for i in text_ngram:
            probability = self.prob(i[0], i[1])
            if probability == 0:
                return float('inf')
            sum_logs += -math.log(probability)

        return math.pow(sum_logs, -(1 /len(text_ngram))) 

################################################################################
# Part 2: N-Gram Model with Interpolation
################################################################################

class NgramModelWithInterpolation(NgramModel):
    ''' An n-gram model with interpolation '''

    def __init__(self, n, k):
        pass

    def get_vocab(self):
        pass

    def update(self, text):
        pass

    def prob(self, context, char):
        pass

################################################################################
# Part 3: Your N-Gram Model Experimentation
################################################################################

if __name__ == '__main__':
    pass
