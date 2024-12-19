import torch
import numpy as np
import keyboard  # For key press detection
import speech_recognition as sr
import transformers
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from transformers import pipeline

# Load the new Wav2Vec2 emotion model and feature extractor
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er")
emotion_processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-large-superb-er")

# Emotion labels (adapt based on your model's configuration)
emotion_labels = ["neutral", "happy", "angry", "sad", "fear", "surprise", "disgust"]

# Initialize the recognizer for SpeechRecognition
recognizer = sr.Recognizer()
def analyze_text_emotions(text):
    pipe = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    results = pipe(text, top_k=None)
    top_3_results = sorted(results, key=lambda x: x['score'], reverse=True)[:3]
    return [(res['label'], res['score']) for res in top_3_results]

def detect_voice_emotions(audio_data, sample_rate=16000):
    inputs = emotion_processor(audio_data, sampling_rate=sample_rate, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = F.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 3)
    return [(emotion_labels[i], top_probs[0][idx].item()) for idx, i in enumerate(top_indices[0])]

def listen_for_commands():
    """
    Listens for voice input, performs speech-to-text, detects emotions, and formats output.
    """
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        # print("Press and hold the Spacebar to speak. Press 'Esc' to exit.")
        while True:
            if keyboard.is_pressed("space"):
                # print("Listening for command...")
                try:
                    # Record audio
                    audio = recognizer.listen(source, timeout=2, phrase_time_limit=20)
                    command_text = recognizer.recognize_google(audio)

                    # Convert audio to float for voice emotion detection
                    raw_audio_data = audio.get_raw_data()
                    audio_data_np = np.frombuffer(raw_audio_data, np.int16).astype(np.float32) / 32768.0

                    # Voice emotion analysis using the new model
                    voice_emotions = detect_voice_emotions(audio_data_np)
                    voice_emotion_output = ", ".join([f"{label} ({prob:.2f})" for label, prob in voice_emotions])

                    # Text-based emotion analysis
                    text_emotions = analyze_text_emotions(command_text)
                    text_emotion_output = ", ".join([f"{label} ({score:.2f})" for label, score in text_emotions])

                    # Final formatted output
                    final_output = (
                        f"{command_text}. "
                        f"The person's speech patterns seem to indicate {voice_emotion_output}, "
                        f"and the sentiment behind the person's text is {text_emotion_output}."
                    )

                    # Print and return the final output
                    # print("Final Output:")
                    print(final_output)

                except sr.UnknownValueError:
                    print("Could not understand the audio.")
                except sr.RequestError as e:
                    print(f"Speech Recognition error: {e}")
                except Exception as e:
                    print(f"An error occurred: {e}")
                # finally:
                #     print("Release the Spacebar to speak again or press 'Esc' to exit.")

            if keyboard.is_pressed("esc"):
                # print("Exiting...")
                break

def main():
    """
    Main entry point for the program.
    """
    listen_for_commands()

if __name__ == "__main__":
    main()
