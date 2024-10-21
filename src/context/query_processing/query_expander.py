from typing import List
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import wordnet

class QueryExpander:
    def __init__(self, word_vectors_path: str):
        self.word_vectors = KeyedVectors.load_word2vec_format(word_vectors_path, binary=True)
        # Check if nltk components: wordnet, averaged_perceptron_tagger is already downloaded
        try:
            nltk.data.find('corpus/wordnet')
        except LookupError:
            print("Downloading 'wordnet' for for NLTK...")
            nltk.download('wordnet', quiet=True)
        try:
            nltk.data.find('tokenizers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading 'averaged_perceptron_tagger' for for NLTK...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
    
    def expand_query(self, query: str, num_expansions: int = 3) -> List[str]:
        original_terms = query.lower().split()
        expanded_terms = set[original_terms]
        for term in original_terms:
            # Word embedding based synonym expansion
            try:
                similar_words = self.word_vectors.most_similar(term, topn=num_expansions)
                expanded_terms.update([word for word, _ in similar_words])
            except KeyError:
                print(f"Word not found in word vectors: {term}")
            # Thesaurus based synonym expansion
            synonyms = wordnet.synsets(term)
            for synonym in synonyms[:2]: # Avoid over-expansion
                expanded_terms.update([lemma.name().replace("_", " ").lower() for lemma in synonym.lemmas()])
        # Remove duplicates and sort
        new_terms = expanded_terms - set(original_terms)
        #combine original query with new terms
        expanded_query = query + " " + " ".join(list(new_terms)[:num_expansions*2])
        return expanded_query

    def expand_query_with_pos(self,query:str, num_expansions:int=3) -> str:
        tokens = nltk.word_tokenize(query)
        pos_tags = nltk.pos_tag(tokens)

        expanded_terms = set(tokens)

        for term,pos in pos_tags:
            term = term.lower()
            # word embedding based synonym expansion
            try:
                similar_words = self.word_vectors.most_similar(term, topn=num_expansions)
                expanded_terms.update([word for word, _ in similar_words])
            except KeyError:
                print(f"Word not found in word vectors: {term}")
            
            # Thesaurus based synonym expansion
            wordnet_pos = self.get_wordnet_pos(pos)
            synonyms = wordnet.synsets(term, pos=wordnet_pos) if wordnet_pos else wordnet.synsets(term)
            for synonym in synonyms[:2]: # Avoid over-expansion
                expanded_terms.update([lemma.name().replace("_", " ").lower() for lemma in synonym.lemmas()])
                    
        # Remove duplicates and sort
        new_terms = expanded_terms - set(tokens)
        #combine original query with new terms
        expanded_query = query + " " + " ".join(list(new_terms)[:num_expansions*2])
        return expanded_query
            
   
    @staticmethod
    def get_wordnet_pos(treebank_tag:str) -> str:
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None