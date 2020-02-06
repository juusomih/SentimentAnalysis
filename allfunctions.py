import nltk
import numpy as np

## Pos tags a list of texts (list of lists) and removes selected word types (depending on function used)
def pos_tagger(list_of_texts): 
    tagged = []
    for text in list_of_texts:
        tagged.append(remove_postags(remove_adv(remove_verb(remove_adj(nltk.pos_tag(text)))))) #Modify what word categories you want to remove here
        
    return tagged ## output is list of texts without the word type


def is_not_adj(tag):
    if tag != "JJ" and tag != "JJR" and tag != "JJS":
        return True
    else: 
        return False

def is_not_noun(tag):
    if tag != "NN" and tag != "NNS" and tag != "NNP" and tag != "NNPS" and tag != "NN$" and tag != "NNS$":
        return True
    else: 
        return False
def is_not_adv(tag):
    if tag != "RB" and tag != "RBR" and tag != "RBT" and tag != "RN" and tag != "RP" and tag != "QL" and tag != "QLP" and tag != "RBS":
        return True
    else: 
        return False
def is_not_verb(tag):
    if tag != "VB" and tag != "VBD" and tag != "VBG" and tag != "VBN" and tag != "VBZ":
        return True
    else: 
        return False
    
def remove_adj(taggedlist): ## Removes adjectives 
    edited_text = [(token,tag) if is_not_adj(tag) else ("<DEL>",tag) for token,tag in taggedlist]
    return edited_text

def remove_adj_adv(taggedlist):
    edited_text = [(token,tag) if is_not_adj(tag) and is_not_adv(tag) else ("<DEL>",tag) for token,tag in taggedlist]
    return edited_text
    
def remove_noun(taggedlist): ## Removes nouns
    edited_text = [(token,tag) if is_not_noun(tag) else ("<DEL>",tag) for token,tag in taggedlist]
    return edited_text

def remove_adv(taggedlist): ## Removes adverbs
    edited_text = [(token,tag) if is_not_adv(tag) else ("<DEL>",tag) for token,tag in taggedlist]
    return edited_text

def remove_verb(taggedlist): ## Removes verbs
    edited_text = [(token,tag) if is_not_verb(tag) else ("<DEL>",tag) for token,tag in taggedlist] 
    return edited_text

def remove_verb_noun(taggedlist): ## Removes verbs
    edited_text = [(token,tag) if is_not_verb(tag) and is_not_noun(tag) else ("<DEL>",tag) for token,tag in taggedlist] 
    return edited_text



def remove_postags(taggedlist): ## Removes postags and outputs the normal text
    edited_text = [token for token,tag in taggedlist]
    return edited_text




def texter(textset ,text_id,id_to_word): #Prints out a single chosen text from the set
    print(' '.join(id_to_word[id] for id in textset[text_id]))
    return None

def id_to_text(indexreview,id_to_word): #Makes text readable from index and into a list of words
    wordreview = []
    for id in indexreview:
        wordreview.append(id_to_word[id])
    return wordreview

def text_to_id(textreview,word_to_id): #Morphs text back from text to indexes and returns it as numpy array format
    root = []
    indexes = []
    for review in textreview:
        if len(indexes) != 0:
            root.append(indexes)
            indexes = []
        for word in review:
            if word not in word_to_id: 
                indexes.append(2)
            else:
                if word_to_id[word] < 9999:
                    indexes.append(word_to_id[word]) 
                else:
                    indexes.append(2)
    root.append(indexes) #needed so root is not 999 long out of a 1000
    final = np.array([np.array(i) for i in root])
    return final

def set_to_text(indexset,id_to_word): ##Makes all indexed text into readable text, list of lists format
    fulltext = []
    for reviews in indexset: 
        fulltext.append(id_to_text(reviews,id_to_word))
    return fulltext










def print_sorted_words(word, metric='cosine'):
    #https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/20_Natural_Language_Processing.ipynb
    layer_embedding = model.get_layer('layer_embedding')
    weights_embedding = layer_embedding.get_weights()[0]
    """
    Print the words in the vocabulary sorted according to their
    embedding-distance to the given word.
    Different metrics can be used, e.g. 'cosine' or 'euclidean'.
    """
   
    # Get the token (i.e. integer ID) for the given word.
    token = word_to_id[word]
    
    # Get the embedding for the given word. Note that the
    # embedding-weight-matrix is indexed by the word-tokens
    # which are integer IDs.
    embedding = weights_embedding[token]
    
    # Calculate the distance between the embeddings for
    # this word and all other words in the vocabulary.
    distances = cdist(weights_embedding, [embedding],
                      metric=metric).T[0]
    
    # Get an index sorted according to the embedding-distances.
    # These are the tokens (integer IDs) for words in the vocabulary.
    sorted_index = np.argsort(distances)
   
    # Sort the embedding-distances.
    sorted_distances = distances[sorted_index]
    
    # Sort all the words in the vocabulary according to their
    # embedding-distance. This is a bit excessive because we
    # will only print the top and bottom words.
    sorted_words = [inverse_map[token] for token in sorted_index if token != 0]

    # Helper-function for printing words and embedding-distances.
    def _print_words(words, distances):
        for word, distance in zip(words, distances):
            print("{0:.3f} - {1}".format(distance, word))

    # Number of words to print from the top and bottom of the list.
    k = 10

    print("Distance from '{0}':".format(word))

    # Print the words with smallest embedding-distance.
    _print_words(sorted_words[0:k], sorted_distances[0:k])

    print("...")

    # Print the words with highest embedding-distance.
    _print_words(sorted_words[-k:], sorted_distances[-k:])
