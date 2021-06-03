import sys
import imageio
import speech_recognition as sr
from tkinter import *
from PIL import ImageTk, Image
from pathlib import Path
import pickle

# list of words to search for in user input for negating classification result
negated_words = ['dont', 'doesnt', 'wasnt', 'cant', 'shouldnt', 'wouldnt', 'couldnt', 'wont', 'not']


# Initialize the variables for playing media
# Code below adopted from https://www.develog.net/2018/09/19/open-and-show-videos-in-tkinter/
pl_vid_name = str(Path().cwd().joinpath('../media', 'pleasant_vid.mp4')) # + '\\media\\pleasant_vid.mp4'
pl_video = imageio.get_reader(pl_vid_name)
pl_data = pl_video.get_meta_data()
pl_delay = int(1000 / pl_data['fps'])
pl_frame = 0

unpl_vid_name = str(Path().cwd().joinpath('../media', 'unpleasant_vid.mp4')) # + '\\media\\unpleasant_vid.mp4'
unpl_video = imageio.get_reader(unpl_vid_name)
unpl_data = unpl_video.get_meta_data()
unpl_delay = int(1000 / unpl_data['fps'])
unpl_frame = 0


def load_model(arg):
    if arg == 'nb':
        model = 'nb_classifier_saved_model.sav'
        counter = 'nb_countvectorizer_saved_counter.sav'
    elif arg == 'knn':
        model = 'nb_classifier_saved_model.sav'
        counter = 'nb_countvectorizer_saved_counter.sav'
    else:
        print("No model preference detected\nUse 'pyhton3 main.py nb' or 'python3 main.py knn'\nTerminating...\n")
        sys.exit(1)

    c = pickle.load(open(Path.cwd().joinpath('../saved_models', model), 'rb'))
    t = pickle.load(open(Path.cwd().joinpath('../saved_models', counter), 'rb'))

    return c, t

# Function to reset the mp4 video frames for looping and when a new classification occurs
def reset_videos():
    global pl_video
    global unpl_video
    pl_video = imageio.get_reader(pl_vid_name)
    unpl_video = imageio.get_reader(unpl_vid_name)


# Function to negate model classification if user input contains a word in the negating list
def negate_prediction(pred):
    if pred[0] == 'pleasant':
        return ['unpleasant']
    else:
        return ['pleasant']


# Function to display pleasant mp4 video in tkinter frame
# Code adopted from https://www.develog.net/2018/09/19/open-and-show-videos-in-tkinter/
def pl_stream(label):
    global play, pl_frame
    try:
        image = pl_video.get_next_data()
        # Keep track of current frame and reset when last frame
        pl_frame += 1
        if pl_frame > (pl_data['fps']*pl_data['duration'])-5:
            reset_videos()
            pl_frame = 0
    except:
        pl_video.close()
        return
    play = label.after(pl_delay, lambda: pl_stream(label))
    frame_image = ImageTk.PhotoImage(Image.fromarray(image))
    label.config(image=frame_image)
    label.image = frame_image


# Function to display unpleasant mp4 video in tkinter frame
# Code adopted from https://www.develog.net/2018/09/19/open-and-show-videos-in-tkinter/
def unpl_stream(label):
    global play, unpl_frame
    try:
        image = unpl_video.get_next_data()
        # Keep track of current frame and reset on last frame
        unpl_frame += 1
        if unpl_frame > (unpl_data['fps'] * unpl_data['duration'])-5:
            reset_videos()
            unpl_frame = 0
    except:
        unpl_video.close()
        return
    play = label.after(unpl_delay, lambda: unpl_stream(label))
    frame_image = ImageTk.PhotoImage(Image.fromarray(image))
    label.config(image=frame_image)
    label.image = frame_image


# Function linked to tkinter button to retrieve text input from user
def get_input():
    global media_label, play, cur_video
    phrase = text_box.get()
    if '\'' in phrase:
        phrase = phrase.replace('\'', '')

    print(phrase)
    doc = [phrase.lower()]

    print("Transforming ", doc, " for classification...")
    doc_bow = transformer.transform(doc)

    print("Predicting phrase sentiment...")
    pred = classifier.predict(doc_bow)

    # Determine negation
    for w in negated_words:
        if w in phrase:
            pred = negate_prediction(pred)

    print("Your phrase was --> ", pred, "\n")

    # Check if new input is same class as last input
    if cur_video != pred[0]:
        # Stop current video playing
        media_label.after_cancel(play)
        reset_videos()
        # determine new video
        if pred[0] == 'pleasant':
            media_label.after(pl_delay, lambda: pl_stream(media_label))
            cur_video = "pleasant"
        elif pred[0] == 'unpleasant':
            media_label.after(unpl_delay, lambda: unpl_stream(media_label))
            cur_video = "unpleasant"


# Function linked to tkinter button for obtaining user speech
def begin_listen():
    global media_label, play, cur_video
    on = True
    utterance = ""
    while on:
        # Code below obtained from
        # https://github.com/Uberi/speech_recognition/blob/master/examples/microphone_recognition.py
        with sr.Microphone() as source:
            print("Say something!")
            audio = r.listen(source)

        try:
            # For testing purposes, we're just using the default API key
            # To use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
            # Instead of `r.recognize_google(audio)`
            utterance = r.recognize_google(audio)
            on = False
            print("Google Speech Recognition thinks you said \"" + utterance, "\"")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            utterance = "blank"
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    if '\'' in utterance:
        utterance = utterance.replace('\'', '')

    speech = [utterance.lower()]

    print("\nTransforming ", speech, " for classification...")
    speech_bow = transformer.transform(speech)

    print("Predicting speech sentiment...")
    pred = classifier.predict(speech_bow)

    # Determine negation
    for w in negated_words:
        if w in utterance:
            pred = negate_prediction(pred)

    print("You said something --> ", pred, "\n")

    # Check if new input is same class as last input
    if cur_video != pred[0]:
        # Stop current video playing
        media_label.after_cancel(play)
        reset_videos()
        # Determine new video
        if pred[0] == 'pleasant':
            media_label.after(pl_delay, lambda: pl_stream(media_label))
            cur_video = "pleasant"
        elif pred[0] == 'unpleasant':
            media_label.after(unpl_delay, lambda: unpl_stream(media_label))
            cur_video = "unpleasant"


if __name__ == '__main__':

    # Load the targeted model
    classifier, transformer = load_model(sys.argv[1])

    # Main application window
    app = Tk()

    # App components
    text_box = Entry(app, width=50)
    r = sr.Recognizer()
    input_button = Button(app, text="Submit", command=get_input)
    listener = Button(app, text="Listen", command=begin_listen)
    nb_button = Button(app, text="Naive Bayes", command=get_input)
    knn_button = Button(app, text="KNN", command=begin_listen)
    media_frame = Frame(app, width=700, height=500)

    media_label = Label(media_frame)
    media_label.pack()

    media_frame.grid(row=0, column=0, columnspan=3)
    text_box.grid(row=1, column=0)
    input_button.grid(row=1, column=1)
    listener.grid(row=1, column=2)

    cur_video = ""
    play = media_label.after(pl_delay, lambda: pl_stream(media_label))

    app.mainloop()
