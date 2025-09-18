import os
import shutil
from pathlib import Path
import whisper
import pandas as pd
import torch
import re



def reorganize_data_simple(source_dir, target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    # 用于记录文件信息的列表
    file_info = []
    for serial_folder in os.listdir(source_dir):
        serial_path = os.path.join(source_dir, serial_folder)
        if os.path.isdir(serial_path):
            for label in os.listdir(serial_path):
                label_path = os.path.join(serial_path, label)
                if os.path.isdir(label_path) and label in ['ad', 'cn']:
                    target_label_dir = os.path.join(target_dir, label)
                    Path(target_label_dir).mkdir(parents=True, exist_ok=True)
                    for file in os.listdir(label_path):
                        if file.endswith('.wav'):
                            # 解析文件名中的序号
                            filename_without_ext = os.path.splitext(file)[0]
                            parts = filename_without_ext.split('_')
                            # 检查是否有足够的部分并且两个序号不相同
                            if len(parts) >= 3:
                                try:
                                    first_id = int(parts[1])
                                    second_id = int(parts[2])

                                    # 如果两个序号相同，则跳过这个文件
                                    if first_id == second_id:
                                        continue
                                except (ValueError, IndexError):
                                    # 如果无法解析序号，也跳过该文件
                                    continue
                            # 复制文件
                            source_file = os.path.join(label_path, file)
                            target_file = os.path.join(target_label_dir, file)
                            shutil.copy2(source_file, target_file)
                            print(f"复制: {source_file} -> {target_file}")

                            # 记录文件信息（去掉扩展名的文件名和标签）
                            file_info.append({'adressfname': filename_without_ext, 'dx': label})
    # 将文件信息保存到CSV文件
    if file_info:
        df = pd.DataFrame(file_info)
        csv_path = os.path.join(target_dir, '..', 'adresso-train-mmse-scores_aug.csv')
        df.to_csv(csv_path, index=False)
        print(f"文件信息已保存至: {csv_path}")
# 增广数据的组织结构：adrso_002_144.wav，第一个序号表示speaker的id，第二个序号表示文本的id，该数据的标签主要由第二个序号来决定的


def remove_non_english(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

# todo 目前是将原始数据与增广数据混合按照标签放进指定目录了，不过为了避免重复编码，只需要将增广数据放进对应目录就行，原始数据还是用之前的处理。
## 因此，需要在完成whisper转录之后，删除第一序号与第二序号相同的样本的asr结果就行，之后再提取对应的音频特征
def preprocess_whisper():
    root_path = '../../dataset/ADReSSo21/diagnosis/train_aug/audio'
    diagnosis = ['ad', 'cn']
    # 这个是asr转录出来的文本要保存的位置
    textual_data = '../../dataset/ADReSSo21/diagnosis/train_aug/text_transcriptions.csv'

    # todo 后续使用whisper large试试，以及后续尝试使用whisper X来提升对齐精度
    # model = whisper.load_model("small.en")
    model = whisper.load_model("large-v3")
    # model = whisper.load_model("large-v3-turbo")
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


if __name__ == "__main__":
    source_directory = "/work/2024/wenbin/Dataset/ADReSSo_Seperate_reverse/enhanced"  # 修改为你的源数据目录
    target_directory = "../../dataset/ADReSSo21/diagnosis/train_aug/audio"  # 修改为你的目标目录
    reorganize_data_simple(source_directory, target_directory)
    # preprocess_whisper()
