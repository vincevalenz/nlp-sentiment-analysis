#  Interactive Application

## Introduction
This is the main file that runs the interactive application for clasifying emotion from text.

To run this application make sure you are in the `/app` directory and then use one the following commands:

&emsp;`python3 main.py nb` to run the application with the naive bayes model

&emsp;`python3 main.py knn` to run the application with the k nearest neighbors model


## Packages
* Imageio
    * `pip install imageio`
    * `pip install imageio-ffmpeg`
* Speech Recognition
    * `pip install SpeechRecognition`
        * This package requires `PyAudio`
* Pillow
    * `pip install Pillow`
* Pathlib
    * `pip install pathlib`
* Tkinter
* Pickle

## Globals
* **negated_words**: List of negated words used for negating an input from a user
* **classifier**: Loaded saved model
* **transformer**: Loaded saved counter
* ***_vid_name**: Path to video
* ***_video**: Video object
* ***_data**: Video meta data
* ***_delay**: Delay used before putting next frame on screen
* ***_frame**: Used to count the number of frames for looping purposes
* **app**: Main application window
* **text_box**: Entry object for user text input
* **r**: Speech recognition object
* **input_button**: Button to retrieve text input from user
* **listener**: Button to invoke speech recognition
* **media_frame**: Frame used to house the video play
* **media_label**: Label to update each video image
* **cur_video**: Current video being played. Used to prevent video restarting on new input
* **play**: Object to hold **media_label** and cancel video play on transition

## Functions
* **load_model(arg)**:
    * Parameters
        * **arg**: String indicating which model to load
    * Functionality
        * Sets the variables **classifer** and **transformer** to the targeted model
    

* **get_input()**:
    * Parameters
        * None
    * Functionality
        * Retrieves text input from the user
        * Processes the input by removing any apostrophes and making all lowercase
        * Sets the negation value to 1 if a negating word exists in the text
        * Classifies the text
        * Determines the correct media to display, invokes the stream and updates the current video
    

* **begin_listen()**:
    * Parameters
        * None
    * Functionality
        * Begins listening on microphone 
        * Processes the result by removing any apostrophes and making all lowercase
        * Sets the negation value to 1 if a negating word exists in the text
        * Classifies the text
        * Determines the correct media to display, invokes the stream and updates the current video
    

* **pl_stream(label)**:
    * Parameters
        * **label**: tkinter label
    * Functionality
        * Displays the images of the pleasant video in the window
        * Tracks the current frame of the video and if it is the last one, it will invoke **reset_videos()**
    

* **unpl_stream(label)**:
    * Parameters
        * **label**: tkinter label
    * Functionality
        * Displays the images of the unpleasant video in the window
        * Tracks the current frame of the video and if it is the last one, it will invoke **reset_videos()**
    

* **reset_videos()**:
    * Parameters
        * None
    * Functionality
        * Resets the image frames of both the pleasant and unpleasant videos
    

## Refernces
* Tkinter Documentation
    * [Tkdocs.com](https://tkdocs.com/)
* Speech Recognition
    * [Python Documentation](https://pypi.org/project/SpeechRecognition/)
    * [Github code](https://github.com/Uberi/speech_recognition/blob/master/examples/microphone_recognition.py)
* Media Display
    * [Develog Blog](https://www.develog.net/2018/09/19/open-and-show-videos-in-tkinter/)
    * [Free MP4 videos](https://www.videezy.com/free-video/mp4-videos)