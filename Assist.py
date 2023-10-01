import speech_recognition as sr
import pyttsx3
import os
import openai
from dotenv import load_dotenv
import playsound
import time

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# initialize the recognizer
recognizer = sr.Recognizer()

# female voice engine initialize
engine = pyttsx3.init()

openai.api_key = api_key

with sr.Microphone() as source:
    print("Listening for 'Hey Jarvis'...")

    while True:
        # audio = recognizer.listen(source)
        audio = "hey jarvis"
        try:
            # text = recognizer.recognize_google(audio)
            text = "hey jarvis"
            if "hey jarvis" in text.lower():
                print("Wake word detected.")
                # playsound.playsound("audio/wake_word.mp3")
                engine.say("Yes, how can I help you?")
                engine.runAndWait()

                print("Listening for user input...")

                # listen for user input
                while True:

                    # audio = recognizer.listen(source)
                    try:
                        # user_input = recognizer.recognize_google(audio)
                        user_input = "What is photosynthesis?"
                        print("User said: ", user_input)

                        # user input search on open ai
                        # response = openai.Completion.create(
                        #     engine="text-davinci-003",
                        #     prompt=user_input,
                        #     max_tokens=50
                        # )

                        # using gpt-3.5-turbo model
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are \"Roshni\" an AI powered assistant Created by Team Syntax Sorcerers and you help the blind people. You are installed in a smart sunglasses made by the team, you can help users to navigate, calculate, answer queries and calculate."
                                },
                                {
                                    "role": "user",
                                    "content": user_input
                                }
                            ],
                            temperature=1.06,
                            max_tokens=71,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )

                        response_text = response.choices[0].message['content'].strip()
                        print("Response:", response_text)

                        # Convert OpenAI response to speech and play
                        engine.say(response_text)
                        engine.runAndWait()

                        print("Listening for 'Hey Jarvis'...")
                        # go back to listen wake word
                        break

                    except sr.UnknownValueError:
                        print("Sorry, I could not understand what you said.")
                    except sr.RequestError as e:
                        print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            pass

