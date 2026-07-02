import whisper
import os
import pandas as pd
import torch
import random
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def remove_non_english(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

# todo 后续使用whisper large试试，以及后续尝试使用whisper X来提升对齐精度
# model = whisper.load_model("small.en")
model = whisper.load_model("large-v3")
# model = whisper.load_model("large-v3-turbo")

root = os.path.join(PROJECT_ROOT, 'datasets', 'ADReSSo21', 'diagnosis', 'train')
root_path = os.path.join(root, 'audio')
diagnosis = ['ad', 'cn']
textual_data = os.path.join(root, 'text_transcriptions.csv')

# textual_data = os.path.join(root, 'text_transcriptions_large.csv')
# textual_data = os.path.join(root, 'text_transcriptions_large_turbo.csv')


def preprocess_whisper():

    df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    for diagno in diagnosis:

        diagno_path = os.path.join(root_path, diagno)

        for file in os.listdir(diagno_path):
            ###兼容wav/mp3
            if file.endswith(".wav") or file.endswith(".mp3"):
                print('Processing:', file)

                # audio_path = os.path.join(diagno_path, file)
                # word_level_path = audio_path.replace('.wav', '.csv').replace('audio', 'text')
                # segmentation_path = audio_path.replace('.wav', '.csv').replace('audio', 'segmentation')
                audio_path = os.path.join(diagno_path, file)
                base = os.path.splitext(audio_path)[0]
                word_level_path = base.replace('audio', 'text') + '.csv'
                segmentation_path = base.replace('audio', 'segmentation') + '.csv'

                excluding_times = []

                if os.path.exists(segmentation_path):
                    df_segmentation = pd.read_csv(segmentation_path)
                    df_segmentation = df_segmentation[df_segmentation['speaker'] == 'INV']
                    for segment in df_segmentation.iterrows():
                        excluding_times += [(segment[1]['begin']/1000, segment[1]['end']/1000)]

                idx_exclude = 0
                result = model.transcribe(audio_path, word_timestamps=True)

                probs = []
                print('Excluding times:', excluding_times)

                transcription = ''
                transcription_pauses = ''
                prev_start = 0.0

                pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])

                for segment in result['segments']:
                    # Print words in segment
                    for word in segment['words']:

                        if idx_exclude < len(excluding_times) and word['start'] >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1

                        if idx_exclude >= len(excluding_times) or word['end'] < excluding_times[idx_exclude][0]:
                            transcription_pauses += word['word']
                            transcription += word['word']
                            clean_word = remove_non_english(word['word'].replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower())
                            if clean_word != '':
                                pandas_word_level = pandas_word_level._append({'word': clean_word, 'start': word['start'], 'end': word['end'], 'probability': word['probability']}, ignore_index=True)
                            probs += [(clean_word, float(word['probability']))]


                            if prev_start > 0.0:
                                pause = word['start'] - prev_start

                                if pause > 2:
                                    transcription_pauses += ' ...'
                                elif pause > 1:
                                    transcription_pauses += ' .'
                                elif pause > 0.5:
                                    transcription_pauses += ' ,'

                            prev_start = word['end']
                        else:
                            print('Excluding word:', word)

                        if idx_exclude < len(excluding_times) and word['end'] >= excluding_times[idx_exclude][1]:
                            idx_exclude += 1

                print('Result:', result['text'])
                print('Transcription:', transcription)
                print('Transcription pauses:', transcription_pauses)
                print('Probs:', probs)

                pandas_word_level.to_csv(word_level_path, index=False)

                df = df._append({
                    'uid': os.path.splitext(file)[0],    # 兼容性改动（MP3）
                    'diagno': diagno,
                    'transcription': remove_non_english(transcription),
                    'transcription_pause': remove_non_english(transcription_pauses),
                    'probablities': probs
                }, ignore_index=True)

    df.to_csv(textual_data, index=False)

preprocess_whisper()
