import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from tkinter import *

lemmatizer = WordNetLemmatizer()
model = load_model('dataset/chatbot_model.h5')

intents = json.loads(open('dataset/intents.json').read())
words = pickle.load(open('dataset/words.pkl', 'rb'))
classes = pickle.load(open('dataset/classes.pkl', 'rb'))

search = {'step': 0,
          'area': '',
          'priceRange': '',
          'typeFood': '',
          'numberPeople': '',
          'restaurantId': '',
          'restaurantName': '',
          'time': '',
          'options': [],
          'data': []}

with open("dataset/restaurants.json", "r") as read_file:
    restaurants = json.load(read_file)


def searchRestaurant():
    food = search['typeFood']

    area = search['area']
    all_zones = ["centre", "north", "south", "east", "west"]
    area = all_zones if area not in all_zones else [area]

    pricerange = search['priceRange']
    pricerange = "cheap" if pricerange == "lo" else "expensive" if pricerange == "hi" else "moderate" if \
        pricerange == "mid" else pricerange

    print("------>>>>>>", food, area, pricerange)
    options = [x for x in filter(lambda x: x["food"] == food and x["area"] in area and x["pricerange"] == pricerange,
                                 restaurants)]

    print(options)
    search['options'] = options
    if len(options) != 0:
        result = "We have found these options: \n"
        optionNumber = 1
        for option in options:
            result += " - " + str(optionNumber) + ": " + option['name'] + ". \n"
            optionNumber += 1
        result += "Which option do you prefer? (Just the number)"
    else:
        result = "We haven't had any result with your search :( Try another  later if you want ;)"
        search['step'] = 0
    return result


def clean_up_sentence(sentence):
    # CLEAN UP SENTENCE WITH TOKENS AND STEMS

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    # RETURN BAG OF WORDS ARRAY: 0 OR 1 FOR EACH WORD IN THE BAG THAT EXISTS IN THE SENTENCE

    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    # PREDICT CLASS OF THE SENTENCE WITH THE MODEL

    # filter our predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json, text):
    # GET RESPONSE FUNCTION

    result = "Copy ;)"
    tag = ints[0]['intent']
    if tag == 'cancel':
        search['step'] = 0
        for i in intents_json['intents']:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    else:
        if search['step'] == 1:
            search['area'] = tag
            search['step'] += 1
            result = "Nice! How much you want to spend? (cheap, moderate, expensive)"
            return result
        elif search['step'] == 2:
            search['priceRange'] = tag
            search['step'] += 1
            result = "Great! Which type of food? (italian, indian, chinese, european, british, mexican, lebanese, " \
                     "international, spanish or french)"
            return result
        elif search['step'] == 3:
            search['typeFood'] = tag
            search['step'] += 1
            result = searchRestaurant()
            return result
        elif search['step'] == 4:
            search['restaurantId'] = search['options'][int(tag) - 1]['id']
            search['restaurantName'] = search['options'][int(tag) - 1]['name']
            search['step'] += 1
            result = "Great! For how many people? (Just the number)"
            return result
        elif search['step'] == 5:
            search['numberPeople'] = tag
            search['step'] += 1
            result = "Great! When you want the table?"
            return result
        elif search['step'] == 6:
            search['time'] = tag
            search['step'] = 0
            search['data'].append({'restaurantName': search['restaurantName'],
                                   'time': search['time'],
                                   'numberPeople': search['numberPeople']})
            result = "Great! Booking a table now at " + search['restaurantName'] + " at " + search['time'] + " for " \
                     + search['numberPeople'] + " people :)"
            return result
        else:
            if tag == 'search':
                result = "Tell me, where you want the restaurant? (centre, east, west, south or north)"
                search['step'] += 1
                return result
            if tag == 'tableBooked':
                if len(search['data']) == 0:
                    result = "You haven't booked a table yet, you can do it now ;)"
                else:
                    print(search['data'])
                    result = "You have this tables booked for today: \n"
                    for data in search['data']:
                        result += " -> Table at " + data['restaurantName'] + " at " + data['time'] \
                                  + " for " + data['numberPeople'] + " people. \n"
                return result

            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    return result

    return result


def chatbot_response(text):
    # SEND FUNCTION FOR BOT

    ints = predict_class(text, model)
    print(ints)
    res = getResponse(ints, intents, text)
    return res


def send():
    # SEND FUNCTION FOR BOT

    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Arial", 12))
        res = chatbot_response(msg)
        ChatLog.insert(END, "CuisineBot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# INTERFACE SET UP
base = Tk()
base.title("CuisineBot by David Valero")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Arial", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#e3da0a", activebackground="#c5bf34", fg='#000000',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")

# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=271, y=401, height=90)
base.mainloop()
