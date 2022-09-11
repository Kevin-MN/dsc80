# project.py


import pandas as pd
import numpy as np
import os
import re
import requests
import time


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    time.sleep(6)
    raw_html = requests.get(url).text

    filter_newline = raw_html.replace('\r\n', '\n')
    start_catch = re.findall(r'\*\*\* START.*\*\*\*((?:[\w]|[^p])*)',filter_newline)
    end_catch = re.sub(r'\*\*\* END(?:[\w]|[^p])*','',start_catch[0])
    return end_catch

# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of every paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of every paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens should include no whitespace.
        - Whitespace (e.g. multiple newlines) between two paragraphs of text 
          should be ignored, i.e. they should not appear as tokens.
        - Two or more newlines count as a paragraph break.
        - All punctuation marks count as tokens, even if they are 
          uncommon (e.g. `'@'`, `'+'`, and `'%'` are all valid tokens).


    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    shake_repl = re.sub(r'\n{2,}',' \x03 \x02 ',book_string)
    shake_repl = re.sub(r'\s{2,}',' ',shake_repl)
    shake_repl = re.sub(r'\n',' ',shake_repl)
    #shake_repl = re.sub(r'.', ' . ')
    tokens = re.findall(r'(?:\\x03|\\x02|\b[\w\d]+\b|[^\w\d\s]+)',shake_repl)
    #re.split(' '|'')

    if tokens[0] == '\x03':
        tokens = tokens[1:]
    elif tokens[0] != '\x02':
        tokens = ['\x02'] + tokens
        
    if tokens[-1] == '\x02':
        tokens = tokens[0:len(tokens)-1]
    elif tokens[-1] != '\x03':
        tokens = tokens + ['\x03']
    return tokens

# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        ser = pd.Series(tokens)
        ser = ser.value_counts()
        return pd.Series(index = ser.index.values, data = np.repeat(1/len(ser.index), len(ser.index)))
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        total_prob = 1
        #test = ('when', 'I', 'drink', 'Coke', 'I', 'smile')
        for n in words:
            if n in self.mdl.index.values:
                total_prob = total_prob * self.mdl.loc[n]
            else:
                total_prob = 0
        return total_prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        return ' '.join(np.random.choice(self.mdl.index.values, size = M,p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        ser = pd.Series(tokens)
        ser = ser.value_counts()
        return pd.Series(index = ser.index.values, data = ser.values/len(tokens))
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        total_prob = 1
        #test = ('when', 'I', 'drink', 'Coke', 'I', 'smile')
        for n in words:
            if n in self.mdl.index.values:
                total_prob = total_prob * self.mdl.loc[n]
            else:
                total_prob = 0
        return total_prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        return ' '.join(np.random.choice(self.mdl.index.values, size = M,p=self.mdl.values))


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        # You don't need to edit the constructor,
        # but you should understand how it works!
        
        self.N = N
        self.tokens = tokens
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            self.prev_mdl = NGramLM(N-1, tokens)

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        #k = 4

        n_grams = []
        for n in np.arange(len(tokens) - self.N + 1):
            n_grams.append(tuple(tokens[n:n+self.N]))
            
        return n_grams
    
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
    
        result = pd.DataFrame(columns = ['ngram','n1gram','prob'])

        ngram_ser = pd.Series(ngrams)
        unique = ngram_ser.unique()
        result['ngram'] = unique
        result['n1gram'] = result['ngram'].apply(lambda x: x[:self.N-1])
        value_countsy = pd.DataFrame(ngram_ser.value_counts()).reset_index()
        value_countsy['index'] = value_countsy['index'].astype(str) 
        value_countsy = value_countsy.set_index('index')
        #val_counts = value_countsy[0]


        #value_countsy.Index = value_countsy.Index.astype(str)
        n1gram_counts = pd.DataFrame(result['n1gram'].value_counts()).reset_index()
        n1gram_counts['index'] = n1gram_counts['index'].astype(str) 
        n1gram_counts = n1gram_counts.set_index('index')
        #n1_counts = n1gram_counts['n1gram']




        result

        probs = []
        for n in unique:
            c_total = 0
            c_n1 = 0
                    

            c_total = value_countsy.loc[str(n)][0]
            c_n1 = n1gram_counts.loc[str(tuple(n[0:self.N-1]))][0]

                    #for r in np.arange(len(self.tokens) - self.N + 1):
                    #    if n == tuple(self.tokens[r:r+self.N]):
                    #        c_total+=1
                    #    if tuple(n[0:self.N-1]) == tuple(self.tokens[r:r+self.N-1]):
                    #        c_n1+=1
                    #for l in np.arange(len(self.tokens) - self.N + 1):
                        
                        

            probs.append(c_total / c_n1 )
                        
                        
        result['prob'] = pd.Series(probs)
        return result
       
            
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4) * (1/2) * (1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        #for n in words:
        #    if n not in self.ngrams:
        #        return 0
    

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        # Use a helper function to generate sample tokens of length `length`
        ...
        
        # Transform the tokens to strings
        ...
        ...
