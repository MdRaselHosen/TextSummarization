import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



chat = "/Users/raselhosen/Desktop/textSummarization/chat.txt"

def extractChat(chat):
    userMessage = []
    AIMessage = []

    with open(chat,'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith("User:"):
                userMessage.append(line[6:].strip())
            elif line.startswith("AI:"):
                AIMessage.append(line[3:].strip())

    return userMessage, AIMessage
userMesssge, AiMessage = extractChat(chat)
print(userMesssge)
print(AiMessage)

# Message Statics
total_message = len(userMesssge) + len(AiMessage)
user_message = len(userMesssge)
Ai_message = len(AiMessage)

print(f'user message count {user_message} AI message counts {Ai_message} Total message {total_message}')
print("******************************")
# Extract top 5 most frequently used words using hashmap

word_frequency = {}
all_messages = userMesssge+AiMessage
for line in all_messages:
    words = line.split()
    for word in words:
        word = word.lower().strip(",.?!")
        if word in word_frequency:
            word_frequency[word]+=1
        else:
            word_frequency[word] = 1


# top five word without removing stop words
top5 = sorted(word_frequency.items(), key=lambda x: x[1],reverse=True)[:5]
for word,value in top5:
    print(f'{word}: {value}')


print("*******************************")
# top 5 word after remove stop words
def top5Words(all_messages):

    words = word_tokenize(all_messages,'english')

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalnum() and word not in stop_words]
    word_count = Counter(words)
    top5 = word_count.most_common(5)

    return top5,words

result = " ".join(all_messages)
top5,words = top5Words(result)
print(top5)
print("words are ", words)
print(all_messages)

print("*****************************")

# Text Summarization


def chat_summary(chat, num_sentences=2):

    sentences = sent_tokenize(chat)

    tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = tfidf.fit_transform(sentences)


    sentence_score = np.sum(tfidf_matrix.toarray(), axis=1)

    top_indices = sentence_score.argsort()[-num_sentences:][::-1]
    top_sentences = [sentences[i] for i in sorted(top_indices)]

    return " ".join(top_sentences)

summary = chat_summary(" ".join(all_messages))
print(summary)
