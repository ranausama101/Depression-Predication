

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
nltk.download('stopwords')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Bidirectional,Embedding,Dropout,LSTM,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("data.csv")
df =df[["message","label"]]
output = ["1","0"]
df=df[df["label"].isin(output)]
df = df.sample(frac=1)
df =df.dropna()
print(df.label.value_counts())



stop_words = stopwords.words('english')
stemmer =  SnowballStemmer('english')

text_cleaning_re =  "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

def preprocess(text, stem=False):
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df['message'] = df.message.apply(lambda x : preprocess(x))


TRAIN_SIZE = 0.8
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 50




tokenize = Tokenizer()

tokenize.fit_on_texts(df['message'])
word_index = tokenize.word_index

vocab_size = len(tokenize.word_index) + 1
print("Vocabulary Size :", vocab_size)

train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE ,random_state=7) # Splits Dataset into Training and Testing set
print("Train Data size:", len(train_data))
print("Test Data size:", len(test_data))

x_train = pad_sequences(tokenize.texts_to_sequences(train_data.message), maxlen=MAX_SEQUENCE_LENGTH)

x_test = pad_sequences(tokenize.texts_to_sequences(test_data.message), maxlen=MAX_SEQUENCE_LENGTH)

# here we will read the glove.6B embeddings and convert it into a dictionary form so we can use it according to our corpus
# creating an embeddy dictiory to store words as keys and feature vector as values
embeddings_dictionary = dict()
# reading the embedding text file
glove_file = open('glove.6B.100d.txt', encoding="utf8")

# looping over each line of the file
# on each line file contains a word and its vector with a space. We will use this structure to read the word with vector
for line in glove_file:
  # spliting the line and getting word and vector
    records = line.split()
    # saving word
    word = records[0]
    # saving the vector and type casting it as array
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    # putting both in the dictionary
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()

# the glove corpus is really big but we want only the words vector which we have in our corpus.for this we only need the selected words from glove dict
#  so here we are going to get that. 

# a embedding_matric 2D array where we are storing all the vector. Here as first arg we are providing vocab_size which is the total no of words in our
# corpus. it will create a 2d array of size equals to our word. 2 arg is 100 which means we want to restrict the vector size to 100 and ignore the rest
embedding_matrix = np.zeros((vocab_size, 100))
#  looping over all the words in our tokenizer, only unique words comes in it
for word, index in tokenize.word_index.items():
  # according to the word we are getting its vector from the embeddings_dictionary we created above using glove. THis returns us the vector of the word
    embedding_vector = embeddings_dictionary.get(word)
    # if the vector is none empty we save it in a new array of embedding_matrix.
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

print("Training X Shape:",x_train.shape)
print("Testing X Shape:",x_test.shape)

labels = train_data.label.unique().tolist()

encoder = LabelEncoder()

y_train = encoder.fit_transform(train_data.label.to_list())
y_test = encoder.fit_transform(test_data.label.to_list())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

EMBEDDING_DIM = 100
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 10


model = Sequential()
# Embedded layer
model.add(Embedding(len(embedding_matrix), EMBEDDING_DIM, weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, trainable=False))

model.add(Bidirectional(LSTM(128,dropout=0.3,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64,dropout=0.3)))
model.add(Dropout(0.3))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))



model.compile(optimizer=Adam(learning_rate=LR), loss='binary_crossentropy',metrics=['accuracy'])

model.load_weights("weights.h5")
# if you want to use the pretrained weights uncomment the above line

# if you want to train uncomment the bottom line.

#model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
#                    validation_data=(x_test, y_test))
#model.save_weights("weights.h5")



#model.evaluate(x_test,yy_test)

tweets = [["haha tweeros like send us pics quotes"],
          ["I'm so upset"],
          ["hiyaaa msn aint working hi"],
          ["nice enough start day bam depression decided time reminder little worth give emoji pensive face"],
          ["course dark side lifestyle dj stress anxiety makes sad people think djs glamorous life struggles anxiety stress depression really push balance"]]

# 1 means depression
# 0 means not depression

prediction = [0,0]
for tweet in tweets:
  t = preprocess(tweet)
  text_test = pad_sequences(tokenize.texts_to_sequences([t]), maxlen=MAX_SEQUENCE_LENGTH)
  a = model.predict(text_test)
  if a[0][0] < 0.50:
    prediction[0] = prediction[0]+1 
  else:
    prediction[1] = prediction[1]+1

if prediction[0]>prediction[1]:
  out = "doesn't have"
else:
  out ="has"


labels = ["non depression","depression"]
graph = plt.pie(prediction,labels=labels,autopct='%1.1f%%')
plt.title("The person {} depression\nNumber of tweets {}".format(out,len(tweets)))
Image = input("Enter Y or N to save the result in image:")
if Image== 'Y':
    plt.savefig('depression_analysis.png')
    print("Image has been saved")
else:
    print("Image is not save:")

