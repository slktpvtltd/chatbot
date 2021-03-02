import nltk
import numpy as np
import string
import nltk

nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def rem_special(word):
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    return(word.translate(remove_punct_dict))

    """
    sentence = "I am sorry! I don't understand you."
    rem_special(sentence)
    print(rem_special(sentence))

    Ans = 'I am sorry I dont understand you'
    """


"""stopword_list = nltk.corpus.stopwords.words('english')


def remove_stopwords(word, is_lower_case=False):
    tokens = tokenizer.tokenize(word)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


sentence1 = "I am sorry! I don't understand you."
remove_stopwords(sentence1)
print(remove_stopwords(sentence1))


# ans = sorry ! ' understand ."""



def bag_of_words(tokenize_sentence, all_words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenize_sentence = [stem(w) for w in tokenize_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0
    return bag
