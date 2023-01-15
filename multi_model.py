import gradio
from pyChatGPT import ChatGPT
from pprint import pprint
import py_secrets
import whisper
import gradio as gr
import time
import warnings

# WHISPER
model = whisper.load_model("base")
model.device

# Transcribe Function
# This function takes in the audio input and converts to text. This text is provided to chatgpt for further processes
def create_chatGPT_api():
    # CHATGPT
    session_token = py_secrets.session_token
    gpt_api = ChatGPT(session_token)
    return gpt_api


def transcribe(audio):
    # Load audio and pad/trim it to fit 30 sec.
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the odel
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    options = whisper.DecodingOptions()
    result= whisper.decode(model,mel,options)
    result_text = result.text

    # passing the generated text to Audio
    gpt_api = create_chatGPT_api()
    resp = gpt_api.send_message(result_text)
    out_result = resp('message')

    return [result_text,out_result]

output_1 = gr.Textbox(label="Speech to Text")
output_2 = gr.Textbox(label="ChatGPT Output")

gr.Interface(
    title = "Integrating OpenAI Whisper and ChatGPT",
    fn = transcribe,
    inputs = [
        gr.inputs.Audio(source='microphone',type ='filepath')
    ],
    outputs = [
        output_1, output_2
    ],
    live =True).launch()

