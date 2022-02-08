import sys
from argparse import ArgumentParser
import nltk, re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.tokenize import SyllableTokenizer
import textstat
from pyphen import Pyphen
# from sentence_transformers import SentenceTransformer, util
import torch
from word2word import Word2word
import stanza
import spacy
import string
from spacy.matcher import Matcher
from tqdm import tqdm
import math
import numpy as np
import collections as coll

PUNCTUATIONS = string.punctuation + '¿¡'

class SingleCorpusLinguisticProperties:
    def __init__(self, corpus):
        # with open(corpus_path, 'r', encoding='utf-8') as f:
        #     lines = f.readlines()
        #     lines = [l.strip('\n') for l in lines]
        # self.corpus = re.sub(r'\b\\u.{4}', '', corpus)
        self.corpus = corpus
        self.sentences = nltk.tokenize.sent_tokenize(self.corpus)
        # self.lines = lines
        # self.lang = lang
        # if lang == 'en':
        self.language = 'english'
        self.spacy_nlp = spacy.load('en_core_web_sm')
        # stanza.download('en')
        self.nlp = stanza.Pipeline(lang='en', processors='tokenize,ner')
        # elif lang == 'de':
        #     self.language = 'german'
        #     self.spacy_nlp = spacy.load('de_core_news_sm')
        #     self.nlp = stanza.Pipeline(lang='de', processors='tokenize')
        # elif lang == 'fr':
        #     self.language = 'french'
        #     self.spacy_nlp = spacy.load('fr_core_news_sm')
        #     self.nlp = stanza.Pipeline(lang='fr', processors='tokenize')
        # elif lang == 'es':
        #     self.language = 'spanish'
        #     self.spacy_nlp = spacy.load('es_core_news_sm')
        #     #self.nlp = stanza.Pipeline(lang='es', processors='tokenize')
        self.pyphen = Pyphen(lang='en')
        # self.model = model
        
        '''
        parsed = []
        if self.lang == 'en' or self.lang == 'de':
            with open('LIdioms/' + self.lang + '/' + self.language + '.ttl', 'r') as f:
                idioms = f.readlines()
                for l in idioms:
                    if l.startswith('liden:'):
                        parsed.append(' '.join(l.split(' ')[0].replace('liden:', '').split('_')))
            self.idiom = parsed
        '''
        self.stops = stopwords.words(self.language)

    def passive_ratio(self):
        print('###passive###')
        passive_count = 0
        matcher = Matcher(self.spacy_nlp.vocab)       
        
        
        for l in tqdm(self.sentences):
            doc = self.spacy_nlp(l)
            passive_rule = [[{'DEP': 'nsubjpass'}, {'DEP': 'aux', 'OP': '*'}, {'DEP': 'auxpass'}, {'TAG': 'VBN'}]]
            matcher.add('Passive', passive_rule)
            matches = matcher(doc)
            if matches:
                passive_count += 1
        return passive_count / len(sentences)

    def comma_count(self):
        print('###comma###')
        num_comma = []
        for l in tqdm(self.sentences):
            count = 0
            doc = self.spacy_nlp(l)
            for token in doc:
                if str(token) in PUNCTUATIONS:
                    count += 1
            num_comma.append(count)
        return sum(num_comma) / len(num_comma)
        
    def sentence_per_sample(self):
        print('###sent###')
        return (len(self.sentences))

    def avg_word_per_sentence(self):
        num_lexicon = 0
        for l in self.sentences:
            num_lexicon += textstat.lexicon_count(l)
        return num_lexicon / len(self.sentences)
    
    def get_vocab_MATTR_density(self):
        corpus = self.corpus
        corpus = re.sub(r'[^\w]', ' ', corpus)
        # corpus = corpus.replace('&apos;', ' ')
        # corpus = corpus.replace('&quot;', ' ')
        corpus = corpus.lower()
        tokens = nltk.tokenize.word_tokenize(corpus, language=self.language)
        types = nltk.Counter(tokens)
        window_size = 50
        ttrs = []
        for i in range(len(tokens) - window_size):
            tok = tokens[i:i + window_size]
            typ = nltk.Counter(tok)
            ttrs.append(100 * len(typ) / window_size)
        MATTR = sum(ttrs) / len(ttrs) # Moving average Type-Token Ratio
        non_stop = [w for w in tokens if not w in self.stops]
        print(len(non_stop) / len(tokens))

        return len(types), MATTR, len(non_stop) / len(tokens)

    def get_syllables(self):
        # pyphen
        # get more precision
        
        corpus = self.corpus
        corpus = re.sub(r'[^\w]', ' ', corpus)
        # corpus = corpus.replace('&apos;', ' ')
        # corpus = corpus.replace('&quot;', ' ')
        corpus = corpus.lower()
        tokens = nltk.tokenize.word_tokenize(corpus, language=self.language)
        SSP = SyllableTokenizer()
        ssp_cache = {}
        syllables = []
        for t in tokens:
            c = ssp_cache.get(t)
            if c is None:
                ssp_cache[t] = len(SSP.tokenize(t))
                syllables.append(ssp_cache[t])
            else:
                syllables.append(c)

        return syllables

    def get_difficult_word_ratio(self):
        difficult = 0
        syllables = self.get_syllables()
        for s in syllables:
            if s >= 3:
                difficult += 1
        return difficult / len(syllables)

    def avg_syllable(self):
        syllables = self.get_syllables()
        return sum(syllables) / len(syllables)

    def get_readability(self):
        corpus = self.corpus
        corpus = corpus.replace('&apos;', ' ')
        corpus = corpus.replace('&quot;', ' ')
        corpus = corpus.replace('\n', ' ')
        # syllables = self.get_syllables()
        avg_syllable = self.avg_syllable()
        textstat.set_lang("en")
        avg_sentence_len = textstat.avg_sentence_length(corpus)
        print(206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllable)
        return 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllable
        
    def RemoveSpecialCHs(self):
        corpus = word_tokenize(self.corpus)
        st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
            "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

        words = [word for word in corpus if word not in st]
        return words
    
    def hapaxLegemena(self):
        words = self.RemoveSpecialCHs()
        V1 = 0
        # dictionary comprehension . har word kay against value 0 kardi
        freqs = {key: 0 for key in words}
        for word in words:
            freqs[word] += 1
        for word in freqs:
            if freqs[word] == 1:
                V1 += 1
        N = len(words)
        V = float(len(set(words)))
        R = 100 * math.log(N) / max(1, (1 - (V1 / V)))
        h = V1 / N
        print(h)
        print(R)
        return R, h
    
    def typeTokenRatio(self):
        words = word_tokenize(self.corpus)
        return len(set(words)) / len(words)

    def BrunetsMeasureW(self):
        words = self.RemoveSpecialCHs()
        a = 0.17
        V = float(len(set(words)))
        N = len(words)
        B = (V - a) / (math.log(N))
        print(B)
        return B

    def YulesCharacteristicK(self):
        words = self.RemoveSpecialCHs()
        N = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        vi = coll.Counter()
        vi.update(freqs.values())
        M = sum([(value * value) * vi[value] for key, value in freqs.items()])
        K = 10000 * (M - N) / math.pow(N, 2)
        return K

    def ShannonEntropy(self):
        words = self.RemoveSpecialCHs()
        lenght = len(words)
        freqs = coll.Counter()
        freqs.update(words)
        arr = np.array(list(freqs.values()))
        distribution = 1. * arr
        distribution /= max(1, lenght)
        import scipy as sc
        H = sc.stats.entropy(distribution, base=2)
        # H = sum([(i/lenght)*math.log(i/lenght,math.e) for i in freqs.values()])
        return H

    def language_detection(self):
        language = set()
        doc = self.nlp(self.corpus)
        for sent in doc.sentences:
            for ent in sent.ents:
                if ent.type == "LANGUAGE" or ent.type == "NORP" or ent.type == "GPE":
                    # if ent.text.lower() != "english":
                    language.add(ent.text)
                
        return language

    def top_100_universities(self):
        import json
        universities = []
        with open("THE_ranking.json") as f:
            data = json.load(f)
            i = 0
            for each in data:
                if i > 100:
                    break
                universities.append(each["name"])
                i += 1
        return universities

    def cosine_match(self, str1, str2):
        from nltk.tokenize import RegexpTokenizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.feature_extraction.text import CountVectorizer
        corpus = [str1, str2]
        token = RegexpTokenizer(r'[a-zA-Z0-9]+')
        count_vectorizer = CountVectorizer(ngram_range=(1,1),tokenizer=token.tokenize, stop_words='english')
        # if corpus:
        #     return False
        vec = count_vectorizer.fit_transform(corpus)
        score = cosine_similarity(vec, vec)[0][1]
        # print(score)
        if score > 0.80:
            return True
        return False


    def institution_rank(self, organizations):
        universities = self.top_100_universities()
        for j in range(len(organizations)):
            if organizations[j]: # if it is not None
                max_score = 0
                for i in range(100):
                    if(self.cosine_match(organizations[j], universities[i])):
                        score = 1 - i/100
                        if (score > max_score):
                            max_score = score
                if (max_score != 0):
                    organizations[j] = max_score
        # print(organizations)
        return organizations


    
if __name__ == '__main__':
    corpus = "Our long-term interest is in machines that contain large amounts of general and scientific knowledge, stored in a \"computable\" form that supports reasoning and explanation. As a medium-term focus for this, our goal is to have the computer pass a fourth-grade science test, anticipating that much of the required knowledge will need to be acquired semi-automatically. This paper presents the first step towards this goal, namely a blueprint of the knowledge requirements for an early science exam, and a brief description of the resources, methods, and challenges involved in the semiautomatic acquisition of that knowledge. The result of our analysis suggests that as well as fact extraction from text and statistically driven rule extraction, three other styles of AKBC would be useful: acquiring definitional knowledge, direct \u201creading\u201d of rules from texts that state them, and, given a particular representational framework (e.g., qualitative reasoning), acquisition of specific instances of those models from text (e..g, specific qualitative models). The house will be cleaned by me every Saturday."
    # corpus = "Global energy and water balance: Characteristics from Finite\u2010volume Atmospheric Model "
    
    corpus = "In this paper, we target on revisiting Chinese pre-trained language models to examine their effectiveness in a non-English language and release the Chinese pre-trained language model series to the community"
    corpus = "The advent of natural language understanding (NLU) benchmarks for English, such as GLUE and SuperGLUE allows new NLU models to be evaluated across a diverse set of tasks. These comprehensive benchmarks have facilitated a broad range of research and applications in natural language processing (NLP). The problem, however, is that most such benchmarks are limited to English, which has made it difficult to replicate many of the successes in English NLU for other languages. To help remedy this issue, we introduce the first large-scale Chinese Language Understanding Evaluation (CLUE) benchmark. CLUE is an open-ended, community-driven project that brings together 9 tasks spanning several well-established single-sentence/sentence-pair classification tasks, as well as machine reading comprehension, all on original Chinese text. To establish results on these tasks, we report scores using an exhaustive set of current state-of-the-art pre-trained Chinese models (9 in total). We also introduce a number of supplementary datasets and additional tools to help facilitate further progress on Chinese NLU. "
    sclp = SingleCorpusLinguisticProperties(corpus)
    sclp.ShannonEntropy()