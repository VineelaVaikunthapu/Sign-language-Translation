import streamlit as st
import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import string
import cv2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import load_model
import os
import threading
from gtts import gTTS
import tensorflow as tf
# Function to recognize speech and display images
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Language in which you want to convert
language = 'en'



page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://www.ixionholdings.com/wp-content/uploads/2021/04/Deaf-people-signing-scaled.jpg");
background-size: cover;

}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("https://www.wallpaperflare.com/static/941/651/764/texture-abstract-white-blue-wallpaper.jpg");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#add_bg_from_url() 

# Load the sign language alphabet prediction model

#class names 
dict_labels = {0:"a",1:"b",2:"c",3:"d",4:"e",5:"f",6:"g",7:"h",8:"i",9:"j",10:"k",11:"l",12:"m",13:"n",14:"o",15:"p",16:"q",17:"r",18:"s",19:"t",20:"u",21:"unknown",22:"v",23:"w",24:"x",25:"y",26:"z"}

# Function to preprocess the image
language = 'en'

# Load trained model's graph
with tf.io.gfile.GFile("trained_model_graph.pb", 'rb') as f:
    # Define a tensorflow graph
    graph_def = tf.compat.v1.GraphDef()

    # Read and import line by line from the trained model's graph
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
# Load training labels file
label_lines = [line.rstrip() for line in tf.io.gfile.GFile("training_set_labels.txt")]
# Function to predict sign language letter
def predict(image_data, sess, softmax_tensor):
    resized_image = image_data[70:350, 70:350]
    resized_image = cv2.resize(resized_image, (200, 200))
    image_data = cv2.imencode('.jpg', resized_image)[1].tostring()
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    max_score = 0.0
    res = ''
    for node_id in top_k:
        if label_lines[node_id].upper() == 'Z':
            human_string = label_lines[node_id+1]
        else:
            human_string = label_lines[node_id]
        score = predictions[0][node_id]
        if score > max_score:   
            max_score = score
            res = human_string
    return res

# Function to speak letter
from gtts import gTTS
import tempfile
import pygame
import time

def speak_letter(letter):
    # Create the text to be spoken
    prediction_text = letter
    
    # Create a speech object from text to be spoken
    speech_object = gTTS(text=prediction_text, lang=language, slow=False)

    # Save the speech object in a temporary file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        speech_object.save(tmp_file.name)
        tmp_file.close()
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Load the speech file
        pygame.mixer.music.load(tmp_file.name)
        
        # Play the speech
        pygame.mixer.music.play()
        
        # Wait for speech to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

# Streamlit app
def main():
    st.title('SIGN-TEXT-SPEECH AND SPEECH-TEXT-SIGN LANGUAGE TRANSLATOR')
    st.sidebar.markdown("# Choose Your Translator")
    activities = ["Speech-Text-Sign", "Sign-Text-Speech"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    #selected_option = st.sidebar.radio("Select Option", ["Detect Live Webcam", "Capture Audio"])

    if choice == "Speech-Text-Sign":

        def func():
            r = sr.Recognizer()
            isl_gif=['all the best', 'any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
            'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
            'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
            'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
            'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
             'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
            'i had to say something but I forgot', 'i have a headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
            'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call an ambulance', 'please call me later',
            'please clean the room', 'please give me your pen', 'please use the dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
            'shall we go together tomorrow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
            'what are you doing', 'what is the problem', 'what is today\'s date', 'what is your age', 'what is your father do', 'what is your job',
            'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when will we go', 'where do you stay',
            'where is the bathroom', 'where is the police station', 'you are wrong','address','agra','ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
    'bihar','bihar','bridge','cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile','dasara',
    'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
    'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
    'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
    'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
    'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
    'voice', 'wednesday', 'weight']
    
            arr=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r',
    's','t','u','v','w','x','y','z']
    
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=5)
                while True:
    
                    st.markdown("<span style='font-size:30px;color:white;'>Say Something...</span>",unsafe_allow_html=True)
                    audio = r.listen(source)
            
                    try:
                        a = r.recognize_google(audio)
                        st.markdown("<span style='font-size:30px;color:white;'>You said:{}</span>".format(a.lower()),unsafe_allow_html=True)
        
                
                # Remove punctuation
                        a = ''.join(ch for ch in a if ch not in string.punctuation)
                
                        if a.lower() == 'stop':
                            st.markdown("<span style='font-size:30px;color:white;'>Oops! Time To say goodbye</span>",unsafe_allow_html=True)
                
                            break
                
                        elif a.lower() in isl_gif:
                            img = Image.open(f'ISL_Gifs/{a.lower()}.gif')
                            img_resized = img.resize((300, 300))  # Resize the image to a smaller size
                            st.image(img_resized, caption=a.lower())
                    
                        else:
                            for i in range(len(a)):
                                if a[i] in arr:
                                    img_path = f'letters/{a[i]}.jpg'
                                    img = Image.open(img_path)
                                    img_resized = img.resize((300, 300))  # Resize the image to a smaller size
                            
                                    st.image(img_resized, caption=a[i])
                                else:
                                    continue

                    except:
                        st.write("Could not listen")
        st.markdown("<span style='font-size:15px; color:white;'>In today's rapidly evolving world, communication is key. Speech-to-text sign technology serves as a vital bridge, enabling seamless interaction between individuals of diverse communication abilities. Through this innovative tool, spoken words are translated into text and then further into sign language, fostering inclusivity and understanding. Let us embrace such advancements, ensuring that no one is left out of the conversation.</span>",unsafe_allow_html=True)
        st.markdown("<span style='font-size:30px;color:white;'>Click the button below and speak into the microphone.</span>",unsafe_allow_html=True)
        if st.button("Record"):
            func()

    else:
        st.markdown("<span style='font-size:15px; color:white;'>Sign-text-speech technology facilitates communication by converting sign language into text and then into spoken words. This innovative tool benefits individuals who are deaf or hard of hearing, allowing them to express themselves more effectively in various settings. By bridging communication barriers, sign-text-speech technology promotes inclusivity and enhances the accessibility of information for all. Its development represents a significant step towards creating a more inclusive and equitable society.</span>",unsafe_allow_html=True)
        st.markdown("<span style='font-size:30px;color:white;'>Click the button below to capture live Signs from your webcam.</span>",unsafe_allow_html=True)
        if st.button("Capture Live Image"):
        # Create a video capture object

        # Display the predicted label
        #st.write(f'Predicted label: {predicted_label}')

        # Get a live stream from the webcam
            live_stream = cv2.VideoCapture(0)

        # Word for which letters are currently being signed
            current_word = ""

            with tf.compat.v1.Session() as sess:
            # Feed the image_data as input to the graph and get the first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

            # Initialize time counter
                time_counter = 0
                captureFlag = False
            # Infinite loop
                while True:
                # Increment time counter
                    time_counter += 1

                # Read a single frame from the live feed
                    ret, img = live_stream.read()

                # Display the live video capture window
                    cv2.imshow("Live Stream", img)

                # Get keypress event
                    keypress = cv2.waitKey(1)

                # To get time intervals
                    if time_counter % 45 == 0:
                        letter = predict(img, sess, softmax_tensor)

                        if letter.upper() != 'NOTHING' and letter.upper() != 'SPACE' and letter.upper() != 'DEL':
                            current_word += letter.upper()
                            speak_letter(letter)

                    # Say the letter out loud
                        elif letter.upper() == 'SPACE':
                            if len(current_word) > 0:
                                speak_letter(current_word)
                            current_word = ""

                        elif letter.upper() == 'DEL':
                            if len(current_word) > 0:
                                current_word = current_word[:-1]

                    # Print the predicted letter for each iteration
                    #st.write("Predicted Letter:", letter.upper())
                        st.markdown("<span style='font-size:30px;'>Predicetd Letter is: {}</span>".format(letter.upper()),unsafe_allow_html=True)
                        st.markdown("<span style='font-size:30px;'>Predicetd Word is: {}</span>".format(current_word.upper()),unsafe_allow_html=True)
                    #st.write("Predicted word:", current_word.upper())
                # 'C' is pressed
                    if keypress == ord('c'):
                        captureFlag = True
                        realTime = False

                # 'R' is pressed
                    if keypress == ord('r'):
                        realTime = True

                    if captureFlag:
                        captureFlag = False

                    # Get the letter and the score
                        letter = predict(img, sess, softmax_tensor)

                        if letter.upper() != 'NOTHING' and letter.upper() != 'SPACE' and letter.upper() != 'DEL':
                            current_word += letter.upper()
                            speak_letter(letter)

                    # Say the letter out loud
                        elif letter.upper() == 'SPACE':
                            if len(current_word) > 0:
                                speak_letter(current_word)
                            current_word = ""

                        elif letter.upper() == 'DEL':
                            if len(current_word) > 0:
                                current_word = current_word[:-1]

                # If ESC is pressed
                    if keypress == 27:
                        break

        # Release the webcam and destroy OpenCV windows
            live_stream.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
