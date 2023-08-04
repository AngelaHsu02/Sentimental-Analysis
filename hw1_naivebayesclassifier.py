from nltk import pos_tag
import random
import math
from collections import defaultdict, Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import string
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import movie_reviews
nltk.download("movie_reviews")
nltk.download('punkt')
# stopwords setting
nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')

import re 
# remove punctuation
punctuation = string.punctuation

# POS tagging
nltk.download("wordnet")
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


# Prepare for train and test data
train_X, train_Y = [], []
test_X, test_Y = [], []
test_fid = []
train_fid = []

random.seed(0)  # 確保每次執行程式時產生的隨機數序列都相同。
for polarity in movie_reviews.categories():
    # for fid in movie_reviews.fileids(polarity)[:5]:
    for fid in movie_reviews.fileids(polarity):
        if random.randrange(5) == 0:
            test_X.append([w for w in movie_reviews.words(fid)])
            test_Y.append(polarity)
            test_fid.append(fid)

        else:
            train_X.append([w for w in movie_reviews.words(fid)])
            train_Y.append(polarity)  # 把類別neg加入train_Y
            train_fid.append(fid)

# Model construciton
class NaiveBayesClassifier:
    def __init__(self, k=0.1):
        self.k = k
        self.features = set()
        self.class_feature_counts = defaultdict(Counter)
        # self.feature_counts = defaultdict(int)
        self.class_counts = Counter()
        self.total = 0
        self.selectedfeatures=[]

    def data_cleaning(self, tokens):
        tokens=re.sub(r'[^\w\s]', '', tokens)
        tokens=re.sub(r'\d', '', tokens)
        tokenize=nltk.word_tokenize(tokens)
        # print(tokenize)
        new_sentence=[t for t in self.lemmatize_with_pos(tokenize) if t not in nltk_stopwords]
        # print(new_sentence)
        return new_sentence
        

    def lemmatize_with_pos(self, tokenize):
        lemmatize = []
        tuples = nltk.pos_tag(tokenize)
        # print(tuples)
        wordnet_lemmatizer = WordNetLemmatizer()
        for tup in tuples:
            pos = self.get_pos_wordnet(tup[1])
            lemma = wordnet_lemmatizer.lemmatize(tup[0], pos=pos)
            # if pos==wordnet.ADJ: #choose trget pos
            lemmatize.append(lemma)
        return lemmatize

    def get_pos_wordnet(self, pos_tag):
        pos_dict = {"N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "J": wordnet.ADJ,
                    "R": wordnet.ADV,
                    "S": wordnet.ADJ_SAT}
        return pos_dict.get(pos_tag[0].upper(), wordnet.NOUN)

    def train(self, train_X, train_Y):
        for tokens, label in zip(train_X, train_Y):
            self.class_counts[label] += 1
            self.total += 1
            tokens=" ".join(tokens)
            # print(tokens)
            new_sentence=self.data_cleaning(tokens)
        # with open(r'C:\Users\USER\Desktop\Python\NLP\new_sentence.txt', 'w') as f:
        #     print(len(new_sentence),new_sentence, file=f)        

            # print(new_sentence)
            for token in set(new_sentence): #iterate new_sentence, only include one notepad              
                self.features.add(token)
                self.class_feature_counts[label][token] += 1
        # with open(r'C:\Users\USER\Desktop\Python\NLP\features.txt', 'w') as f: #print all features
        #     print(len(self.features),self.features, file=f)
        # with open(r'C:\Users\USER\Desktop\Python\NLP\cls_f_cnt.txt', 'w') as f: #print all class feature counts
        #     print(self.class_feature_counts, file=f)
        # print(self.class_feature_counts) #print all class feature counts
        
        for feature in self.features:
            # if self.class_feature_counts[label][feature]>=100:
            if self.class_feature_counts['neg'][feature]>=20 or self.class_feature_counts['pos'][feature]>=20: #freq>=20 85.3% (k=0.1)
                self.selectedfeatures.append(feature)
        # print(len(self.selectedfeatures),self.selectedfeatures)


    def probabilities(self, feature):
        probs = {}
        for cls, cls_cnt in self.class_counts.items():
            probs[cls] = (self.class_feature_counts[cls][feature] +
                          self.k) / (cls_cnt + len(self.class_counts) * self.k)
        return probs

    def predict(self, tokens):
        # print(tokens)
        tokens=" ".join(tokens)
        # print(tokens)
        new_sentence=self.data_cleaning(tokens)
        # with open(r'C:\Users\USER\Desktop\Python\NLP\test.txt', 'w') as f:
        #     print(new_sentence, file=f)
    
        log_probs = Counter()
        for cls, cls_cnt in self.class_counts.items():
            log_probs[cls] = math.log(cls_cnt / self.total) # type: ignore

        for feature in self.selectedfeatures:
            probs = self.probabilities(feature)

            if feature in set(new_sentence):   #predicted review can have replicated words            
                for cls, prob in probs.items():
                    log_probs[cls] += math.log(prob) # type: ignore
            else:
                for cls, prob in probs.items():
                    log_probs[cls] += math.log(1.0 - prob) # type: ignore
        # print(set(new_sentence))
        return max(log_probs, key=log_probs.get), log_probs # type: ignore

model = NaiveBayesClassifier()
model.train(train_X, train_Y)


#Calculate precision
correct, total = 0, 0

for x, y in zip(test_X, test_Y):
    prediction, _ = model.predict(x)
    if prediction == y:
        correct += 1
    total += 1
print("Acc: %d / %d = %g" % (correct, total, correct / total))


#Use the model
from nltk.tokenize import word_tokenize
nltk.download('punkt')
review="""I'm not sure what accomplished director/producer/cinematographer Joshua Caldwell was thinking taking on this project.

This film has got to be the epitome of terrible writing and should be a classroom example of 'what not to do' when writing a screenplay. Why would Joshua take on (clearly) amateur writer Adam Gaines script is beyond me. Even his good directing and excellent cinematography could not save this disaster.

Aside from the super obvious plot holes and very poor story overall, the dragged-out unnecessary dialogue made this film unbearable and extremely boring. The way too long 1h 39min film length felt like 4 hours and I found myself saying "get on with it already, who cares!" when the two leads would just ramble on about nothing relevant. This movie may have been interesting if it was a 30 min short film (which oddly enough is the only minimal writing experience Adam Gaines has).

The acting was decent and Katia Winter was very easy on the eyes to look at, but her chemistry with Simon Quarterman was very unconvincing. Maybe it was the boring dialogue they had that made their chemistry absent.

Even the maybe total of 10 minutes of action scenes were overly dragged out. The rest of the film was primarily useless garbage dialogue with absolutely no point to the story - start to finish.

Don't waste your time with this one. See the trailer, and that's all the good and interesting parts you'll need to see.

This gets a 3/10 strictly for the directing and cinematography."""

print(model.predict(word_tokenize(review.lower())))

#Exploring important features
def prob_class_given_feature(feature, cls, model): #對正評條件機率高的feature 
    probs = model.probabilities(feature)
    return probs[cls] / sum(probs.values())

print(sorted(model.features, key=lambda t: prob_class_given_feature(t, "pos", model), reverse=True)[:30])#lambda variables:function, iterable。lambda是匿名的函式，variables輸入變數名稱:function輸入函式，iterable輸入我們要迭代輸入的對象 
print(sorted(model.features, key=lambda t: prob_class_given_feature(t, "neg", model), reverse=True)[:30])