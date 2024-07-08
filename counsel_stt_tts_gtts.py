import os
from dotenv import load_dotenv
import openai
import pandas as pd
import time
from datetime import datetime
from typing_extensions import override
from openai import AssistantEventHandler, OpenAI
from openai.types.beta.threads import Text, TextDelta
from openai.types.beta.threads.runs import ToolCall, ToolCallDelta
import json
import streamlit as st
import pyaudio
import wave
from gtts import gTTS
from playsound import playsound
import speech_recognition as sr

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)
vector_store_id = "vs_ED6yAeEbz9l6ixFNQDRVQabO"

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt = f.read()

def create_assistant():
    response = client.beta.assistants.create(
        name="Rubis Counselor",
        instructions=prompt,
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
        model="gpt-4o",
        temperature=0.5,
        top_p=1.0          
    )
    return response.id

def create_thread():
    response = client.beta.threads.create()
    return response.id

def add_message_to_thread(thread_id, role, content):
    response = client.beta.threads.messages.create(
        thread_id=thread_id,
        role=role,
        content=content
    )
    return response.id

class EventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.response_content = ""

    @override
    def on_text_created(self, text):
        print(f"\n위스퍼: ", end="", flush=True)

    @override
    def on_text_delta(self, delta, snapshot):
        print(delta.value, end="", flush=True)
    
    @override
    def on_text_done(self, text):
        self.response_content = text.value
        
    @override
    def on_tool_call_created(self, tool_call):
        pass
        # print(f"\ntool>{tool_call.type}\n", flush=True)

def get_response(thread_id, assistant_id):
    event_handler = EventHandler()

    with client.beta.threads.runs.stream(
        thread_id=thread_id,
        assistant_id=assistant_id,
        event_handler=event_handler
    ) as stream:
        stream.until_done()

    return event_handler.response_content

def record_audio(duration, filename):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    WAVE_OUTPUT_FILENAME = filename

    audio = pyaudio.PyAudio()

    # start recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def transcribe_speech(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language='ko-KR')
    except sr.UnknownValueError:
        return "음성을 인식할 수 없습니다."
    except sr.RequestError as e:
        return f"Google 음성 인식 서비스에 접근할 수 없습니다; {e}"

def synthesize_speech(text, filename):
    tts = gTTS(text=text, lang='ko')
    tts.save(filename)
    # playsound(filename)

def main():
    counsel_start_time = time.time()
    counsel_start_time_str = datetime.fromtimestamp(counsel_start_time).strftime('%Y-%m-%d %H:%M:%S')
    
    df = pd.DataFrame(columns=['role', 'text', 'response_latency', 'start_time', 'end_time'])
    latency_df = pd.DataFrame(columns=['task', 'latency'])
    
    # initial_greeting = "안녕하세요, 저는 오늘 함께 이야기를 나눌 위스퍼입니다.\n여러 일들로 마음이 불편하고 자기 감정을 알기 어려울 때 상담을 통해 마음의 안정을 찾도록 돕는 일을 하고 있어요.\n간단하게 자기 소개를 부탁해도 될까요?"
    initial_greeting = "안녕하세요"
    
    print(f"{initial_greeting}")
    
    new_row = pd.DataFrame([{
        'role': 'counselor',
        'text': initial_greeting,
        'response_latency': 'N/A',
        'start_time': counsel_start_time_str,
        'end_time': counsel_start_time_str
    }])
    
    df = pd.concat([df, new_row], ignore_index=True)

    assistant_id = create_assistant()
    thread_id = create_thread()

    add_message_to_thread(thread_id, "assistant", initial_greeting)

    synthesize_speech(initial_greeting, "initial_greeting.mp3")
    # playsound("initial_greeting.mp3")

    while True:
        user_start_time = time.time()
        
        record_audio(10, "user_input.wav")  # 10초 동안 음성을 녹음합니다.

        stt_start_time = time.time()
        user_input = transcribe_speech("user_input.wav")
        stt_end_time = time.time()
        stt_latency = stt_end_time - stt_start_time
        
        print(f"\n내담자: {user_input}")
        
        user_end_time = time.time()

        if user_input.lower() == '종료':
            end_message = "위스퍼: 그럼 상담은 여기까지로 마치도록 하겠습니다.\n좋은 하루 보내세요!"
            print(end_message)
            synthesize_speech(end_message, "end_message.mp3")
            # playsound("end_message.mp3")
            break

        user_row = pd.DataFrame([{
            'role': 'client',
            'text': user_input,
            'response_latency': f"{user_end_time - user_start_time:.1f}초",
            'start_time': datetime.fromtimestamp(user_start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(user_end_time).strftime('%Y-%m-%d %H:%M:%S')
        }])

        df = pd.concat([df, user_row], ignore_index=True)
        
        elapsed_time = time.time() - counsel_start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        elapsed_time_str = f"{minutes}:{seconds:02d}"

        gpt_start_time = time.time()
        add_message_to_thread(thread_id, "user", f"{user_input}, Time elapsed:{elapsed_time_str}")

        # gpt_start_time = time.time()
        response = get_response(thread_id, assistant_id)
        gpt_end_time = time.time()
        gpt_latency = gpt_end_time - gpt_start_time
       
        
        response_time = time.time()
        counselor_row = pd.DataFrame([{
            'role': 'counselor',
            'text': response,
            'response_latency': f"{response_time - user_end_time:.1f}초",
            'start_time': datetime.fromtimestamp(user_end_time).strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': datetime.fromtimestamp(response_time).strftime('%Y-%m-%d %H:%M:%S')
        }])

        df = pd.concat([df, counselor_row], ignore_index=True)
        
        tts_start_time = time.time()
        synthesize_speech(response, "response.mp3")
        tts_end_time = time.time()
        tts_latency = tts_end_time - tts_start_time
        # playsound("response.mp3")
        
        # Calculate end-to-end latency
        e2e_latency = time.time() - user_start_time

        # waiting time = stt + gpt + tts 
        waiting_latency = tts_end_time - stt_start_time
        
        latency_data = [
            {'task': 'stt', 'latency': f"{stt_latency:.1f}초"},
            {'task': 'gpt', 'latency': f"{gpt_latency:.1f}초"},
            {'task': 'tts', 'latency': f"{tts_latency:.1f}초"},
            {'task': 'e2e', 'latency': f"{e2e_latency:.1f}초"},
            {'task': 'waiting', 'latency': f"{waiting_latency:.1f}초"}
        ]

        latency_df = pd.concat([latency_df, pd.DataFrame(latency_data)], ignore_index=True)

        csv_file_path = './records/records.csv'
        df.to_csv(csv_file_path, index=False)
        latency_csv_file_path = './records/latency_records.csv'
        latency_df.to_csv(latency_csv_file_path, index=False)
    
    total_elapsed_time = time.time() - counsel_start_time
    total_minutes, total_seconds = divmod(int(total_elapsed_time), 60)
    print(f"상담 총 시간: {total_minutes}분 {total_seconds}초")

if __name__ == "__main__":
    main()
