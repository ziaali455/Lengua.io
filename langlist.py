import gensim
import pandas as pd
import tkinter as tk
from tkinter import *
import translate
from translate import Translator

spanishTranslator= Translator(to_lang="Spanish")

englishTranslator= Translator(from_lang="spanish",to_lang="english")


root = tk.Tk()
root.title("Lingua.IO")
root.geometry("700x400")

titleLabel = Label(root, text = "🗣 Lingua.IO 🗣", font = ("arial", 36, "bold"),justify=CENTER)
titleLabel.grid(row = 1, column = 0)

label1= Label(root, text = "Let's find some words related to", font = ("arial", 15))
label1.grid(row = 6, column = 0)

searchedW_var=tk.StringVar()

df = pd.read_json("/Users/alizia/Downloads/reviews_Cell_Phones_and_Accessories_5.json", lines=True)
review_text = df.reviewText.apply(gensim.utils.simple_preprocess)

#print(df.shape) #this works
#print(review_text)

model = gensim.models.Word2Vec(
    window = 10,
    min_count = 2,
    workers=6
)

model.build_vocab(review_text, progress_per= 100)
model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)
model.save("./word2vec-amazon-cell-accessories-reviews-short.model")

enter_box = Entry(root, textvariable=searchedW_var, width = 35)
enter_box.grid(row=6, column = 1)


def search():
    searchedWord = searchedW_var.get()
    print("search function works")
    print(searchedWord)
    searchedW_var.set("")
    resultLabel = Label(root, text = "", font = ("arial", 36, "bold"),justify=CENTER)
    resultLabel.grid(row = 5, column=3)
    #print(model.wv.most_similar(searchedWord, topn=100))
    searchedWordEng = englishTranslator.translate(searchedWord)
    just_first = [a for a, b in model.wv.most_similar(searchedWord, topn=100)]
    #print(just_first)
    for x in just_first:
        similarWordsSpan = spanishTranslator.translate(x)
        print (similarWordsSpan)
        
       # resultLabel.config(text=similarWordsSpan)


btn = Button (root, text = "Search", command=search, bg = "blue", fg = "black", font = ("arial", 14, "bold"))
btn.grid(row=6, column=4)

root.mainloop()