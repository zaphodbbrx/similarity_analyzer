#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@InProceedings{polyglot:2013:ACL-CoNLL,
 author    = {Al-Rfou, Rami  and  Perozzi, Bryan  and  Skiena, Steven},
 title     = {Polyglot: Distributed Word Representations for Multilingual NLP},
 booktitle = {Proceedings of the Seventeenth Conference on Computational Natural Language Learning},
 month     = {August},
 year      = {2013},
 address   = {Sofia, Bulgaria},
 publisher = {Association for Computational Linguistics},
 pages     = {183--192},
 url       = {http://www.aclweb.org/anthology/W13-3520}
}
"""
from polyglot.text import Text
from polyglot.detect import Detector
import numpy as np
from polyglot.mapping.embeddings import Embedding
from polyglot.text import Text
from polyglot.mapping.embeddings import Embedding
from polyglot.mapping import CaseExpander, DigitExpander
import nltk
import pymorphy2


class LanguageNotAvailableError(Exception):
    pass

class LanguageNotRecognisedError(Exception):
    pass 

class SimilarityAnalyzer():
    def __init__(self):
        self.__langs = ['en', 'de', 'ru','cn', 'jp', 'es']

    def lang_detect(self, text, threshold = 0.7):
        detector = Detector(text,quiet = True)
        if detector.language.confidence>threshold:
            return detector.language.code
        else:
            raise LanguageNotRecognisedError('Could not recognize the language')
    def DocSimilarity(self, doc1,doc2,lang = None):
        def get_doc_vector(doc):
            doc_t = Text(doc)
            vectors = []
            lang_vectors = {
                'en':"/home/lsm/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2",
                'ru':"/home/lsm/polyglot_data/embeddings2/ru/embeddings_pkl.tar.bz2"
            }
            embeddings = Embedding.load(lang_vectors[lang])
            for word in doc_t.words:
                if word in embeddings:
                    vectors.append(embeddings[word])
                else:
                    embeddings.vocabulary.words.append(word)
                    embeddings.apply_expansion(CaseExpander)
                    if word in embeddings:
                        vectors.append(embeddings[word])
            vec_arr = np.array(vectors)
            return np.apply_along_axis(np.mean, 0, vec_arr)
        if lang == None:
            lang = self.lang_detect(doc1)
        if lang not in self.__langs:
            raise LanguageNotAvailableError('Language detected is not available at the moment')
        vec1 = get_doc_vector(doc1)
        vec2 = get_doc_vector(doc2)
        res = np.linalg.norm(vec1-vec2)
        return {'lang': lang, 'similarity': res}
    
    def ShowSimilarWords(self, word, lang = None):
        def get_pos_tags(words):
            if lang == 'en':
                res = Text(' '.join(words)).pos_tags
            elif lang == 'ru':
                res = [(w, str(pymorphy2.MorphAnalyzer().parse(w)[0].tag).split(',')[0]) for w in words]
            else:
                res = [(w, 'UNKNOWN') for w in words]
            return res
        if lang==None:
            lang = self.lang_detect(word)
        text = Text(word)
        if len(text.words)==1:
            lang_vectors = {
                'en':"/home/lsm/polyglot_data/embeddings2/en/embeddings_pkl.tar.bz2",
                'ru':"/home/lsm/polyglot_data/embeddings2/ru/embeddings_pkl.tar.bz2"
            }
            embeddings = Embedding.load(lang_vectors[lang])
            if word in embeddings:
                neighbors = get_pos_tags(embeddings.nearest_neighbors(word))
            else:
                embeddings.vocabulary.words.append(word)
                embeddings.apply_expansion(CaseExpander)
                if word in embeddings:
                    neighbors = get_pos_tags(embeddings.nearest_neighbors(word))
                else:
                    neighbors = []
            return {'lang': lang, 'neighbors': neighbors}
        else:
            return 'must be single word'
if __name__ == '__main__':
    sa = SimilarityAnalyzer()
    while True:
        doc1 = input()
        doc2 = input()
        lang = sa.lang_detect(doc1)
        print('Language: '+ lang)
        print('Similarity: \n')
        print(sa.DocSimilarity(doc1,doc2, lang))