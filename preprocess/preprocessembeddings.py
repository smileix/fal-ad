import os
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizerFast, BertModel, DistilBertModel, AutoModel, DistilBertTokenizerFast, AutoModelForCausalLM
import torch
import torchaudio
import opensmile
import unicodedata
import librosa
import math
import numpy as np
import gc
from multiprocessing import Pool
import whisper
from transformers import HubertModel
from transformers import Wav2Vec2BertModel, Wav2Vec2FeatureExtractor, AutoProcessor, Wav2Vec2BertProcessor, AutoFeatureExtractor
from torch.cuda.amp import autocast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def process_egemaps_frame(frame_data):
    frame, sr = frame_data
    # 每个进程需要创建自己的 smile 实例
    smile_instance = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    result = smile_instance.process_signal(frame, sr)
    return result.values

bert_path = "/work/shared/plms/bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avaiable: bert, roberta, distil, stella, mistral, qwen

textual_model = 'distil'
# textual_model = 'bert'
# textual_model = 'roberta'
# textual_model = 'stella'
# textual_model = 'deberta'
# textual_model = 'biobert'
# textual_model = 'tinybert'
# textual_model = 'albert'

# audio_model = ''
audio_model = 'wav2vec2'
# audio_model = 'egemaps'
# audio_model = 'mel'
# audio_model = 'whisper'
# audio_model = 'hubert'  # 新增HuBERT选项
# audio_model = 'wavbert'

pauses = True
# pauses = False

pauses_data = '_pauses' if pauses else ''
name_mapping_text = {
    'bert': 'bert',
    'distil': 'distil',
    'roberta': 'roberta',
    'mistral': 'mistral',
    'qwen': 'qwen',
    'stella': 'stella',
    'deberta': 'deberta',  # 添加DeBERTa映射
    'biobert': 'biobert',  # 添加BioBERT映射
    'tinybert': 'tinybert',
    'albert': 'albert',
}
textual_model_data = '_' + name_mapping_text.get(textual_model, '')
name_mapping_audio = {
    'wav2vec2': 'audio',
    'egemaps': 'egemaps',
    'mel': 'mel',
    'whisper': 'whisper',
    'hubert': 'hubert',
    'wavbert': 'wavbert',
}
audio_model_data = '_' + name_mapping_audio.get(audio_model, '')
root = '../../dataset/ADReSSo21/diagnosis/train/'
# root = '../../dataset/ADReSSo21/diagnosis/test-dist/'
root_path = os.path.join(root, 'audio')
root_text_path = os.path.join(root, 'text')
textual_data = os.path.join(root,'text_transcriptions.csv')
max_length = 200
# todo 这里的max length替换成mini batch的做法


def split_audio_with_fixed_duration(waveform, sample_rate, max_chunk_duration=150.0):
    """
    Split waveform into fixed-duration chunks.

    :param waveform: Input audio waveform (1D tensor).
    :param sample_rate: Sampling rate of the audio.
    :param max_chunk_duration: Duration of each chunk in seconds. Default is 60 seconds.
    :return: List of audio chunks and their corresponding start/end times.
    """
    # Convert waveform to numpy array if it's a tensor
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()

    # Calculate the number of samples per chunk
    max_chunk_samples = int(max_chunk_duration * sample_rate)

    # Split the waveform into fixed-size chunks
    chunks = []
    chunk_intervals = []
    num_samples = len(waveform)
    start_sample = 0

    while start_sample < num_samples:
        end_sample = min(start_sample + max_chunk_samples, num_samples)
        chunk = waveform[start_sample:end_sample]
        chunks.append(chunk)

        # Record the interval in samples
        chunk_intervals.append((start_sample, end_sample))

        # Move to the next chunk
        start_sample += max_chunk_samples

    return chunks, chunk_intervals

def preprocess_text():
    if textual_model == 'bert':
        ###做了一定修改使得可以本地运行
        # model_dir = "/work/shared/plms/bert-base-uncased"
        model_dir = "/work/shared/plms/bert-large-uncased"
        tokenizer = BertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
        model = BertModel.from_pretrained(model_dir, local_files_only=True).to(device)
    elif textual_model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base").to(device)
    elif textual_model == 'distil':
        model_dir = "/work/shared/plms/distillbert-base-uncased"
        # model_dir = "/work/shared/plms/distilbert-base-cased"

        tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
        model = DistilBertModel.from_pretrained(model_dir, local_files_only=True).to(device)
    elif textual_model == 'stella':
        tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
        model = AutoModel.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
    elif textual_model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        # Need Access Token
        model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
    elif textual_model == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        # Need Access Token
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    elif textual_model == 'deberta':
        model_dir = "/work/shared/plms/deberta-base"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        # tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    # 在模型加载条件分支中添加BioBERT支持（在elif textual_model == 'deberta':之前添加）
    elif textual_model == 'biobert':
        # BioBERT - 生物医学领域预训练的BERT模型
        biobert_path = "/work/shared/plms/biobert-base-cased-v1.1"  # 根据实际路径修改
        tokenizer = AutoTokenizer.from_pretrained(biobert_path, local_files_only=True)
        model = AutoModel.from_pretrained(biobert_path, local_files_only=True).to(device)
    elif textual_model == 'tinybert':
        model_dir = "/work/shared/plms/tinybert"
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
    elif textual_model == 'albert':
        # Albert - 支持预训练模型或本地路径
        albert_path = "/work/shared/plms/albert-base-v2"  # 根据实际路径修改
        tokenizer = AutoTokenizer.from_pretrained(albert_path, local_files_only=True)
        # tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
        model = AutoModel.from_pretrained(albert_path, local_files_only=True).to(device)

    model.eval()

    # segment_length可以看成是帧数，wav2vec2默认是50帧，也就是1帧表示20ms，egemaps默认是10帧，1帧表示100ms。
    # 先由声学模型提取帧级特征，再根据whisper提取的word level的时间戳（标点符号也会考虑在内），来切片对应时间段的特征，并均值池化获得对应音频特征向量。
    # 以此实现word基本的文本特征与音频特征的对齐
    # 文本层面上，是否建模停顿的区别只是表示停顿的三种符号的添加与否，当停顿被加入时，文本特征向量的长度会变长（多了一些表示停顿的符号），同时音频特征向量在词级对应时也会相应变长
    if audio_model == 'wav2vec2':
        ###做了一定修改使得可以本地运行
        wav2vec_path = "/work/shared/plms/wav2vec2-base-960h/"
        processor = Wav2Vec2Processor.from_pretrained(wav2vec_path, local_files_only=True)
        wav2vec_model = Wav2Vec2Model.from_pretrained(wav2vec_path, local_files_only=True).to(device)
        segment_length = 50
    elif audio_model == 'egemaps':
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals, )
        segment_length = 10
    elif audio_model == 'whisper':
        # todo 后续使用large试试，也可以尝试使用whisperx提升对齐精度
        whisper_model = whisper.load_model("small.en")  # 可以选择 tiny, base, small, medium, large
        segment_length = 50  # 根据 Whisper 的帧率调整
    elif audio_model == 'hubert':  # 添加HuBERT支持
        hubert_path = "/work/shared/plms/hubert-base-ls960"
        wav2vec_path = "/work/shared/plms/wav2vec2-base-960h/"
        processor = Wav2Vec2Processor.from_pretrained(wav2vec_path, local_files_only=True)
        hubert_model = HubertModel.from_pretrained(hubert_path, local_files_only=True).to(device)
        segment_length = 50
    elif audio_model == 'wavbert':
        # WavBert - 音频-文本联合表示模型
        wavbert_path = "/work/shared/plms/wavbert"  # 根据实际路径修改
        # processor = AutoProcessor.from_pretrained("facebook/w2v-bert-2.0")
        # processor = AutoProcessor.from_pretrained(wavbert_path)
        # processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        # processor = Wav2Vec2BertProcessor.from_pretrained("facebook/w2v-bert-2.0")
        # processor = Wav2Vec2Processor.from_pretrained("facebook/w2v-bert-2.0")
        # processor = Wav2Vec2Processor.from_pretrained("facebook/w2v-bert-2.0")
        processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0", tokenizer=None,  # 明确禁用 tokenizer
            feature_extractor="facebook/w2v-bert-2.0")
        w2vbert_model = Wav2Vec2BertModel.from_pretrained(wavbert_path).cuda()
        w2vbert_model.eval()
        segment_length = 50

    else:
        segment_length = 50

    # Read textual data from CSV
    df = pd.read_csv(textual_data, encoding='utf-8')
    row_data = 'transcription_pause' if pauses else 'transcription'
    df[row_data] = df[row_data].apply(lambda x: unicodedata.normalize("NFC", str(x)))
    completed_audios = 0
    # Columns are     df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    # Iteate over each row
    for index, row in df.iterrows():

        print(f"------------------------------------------")
        print(f"Processing {row['uid']}, {row['diagno']}")


        # Get the transcription
        transcription = row[row_data]

        # Tokenize the transcription
        inputs_text = tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Get the embeddings
        with torch.no_grad():
            outputs_text = model(**inputs_text)

        # Save the embeddings
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()
        torch.save(last_hidden_states_text, os.path.join(root_text_path, row['diagno'], row['uid'] + textual_model_data + pauses_data + '.pt'))

        # ...上面文本embedding相关代码...

        if audio_model != '':
            #audio_path = os.path.join(root_path, row['diagno'], row['uid'] + '.wav')
            ###兼容mp3
            audio_path_wav = os.path.join(root_path, row['diagno'], row['uid'] + '.wav')
            audio_path_mp3 = os.path.join(root_path, row['diagno'], row['uid'] + '.mp3')
            if os.path.exists(audio_path_wav):
                audio_path = audio_path_wav
            elif os.path.exists(audio_path_mp3):
                audio_path = audio_path_mp3
            else:
                print(f"ERROR: No audio file found for {row['uid']}")

            # 初始化 processed_audio_tensor，后面所有分支都用到
            processed_audio_tensor = None

            if audio_model == 'wav2vec2':
                # 1. 加载音频并重采样到16kHz
                wave_form, sample_rate = torchaudio.load(audio_path)
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)  # 转mono
                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                # 2. 提取wav2vec2特征
                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = wav2vec_model(**inputs_audio)

                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()

                # 3. 检查/修正NaN
                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)

                # 4. 统一命名和初始化
                features_audio = last_hidden_states_audio
                frame_level_audio = features_audio
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                processed_audio_tensor[0] = features_audio.mean(dim=0)

            elif audio_model == 'egemaps':
                # 1. 加载音频
                y, sr = librosa.load(audio_path)
                frame_size = 0.1  # 100ms一帧
                frame_samples = int(frame_size * sr)
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                # # 这段处理效率太低，需要改进
                # # 2. 提取egemaps特征
                # features = []
                # for frame in frames:
                #     features.append(smile.process_signal(frame, sr))
                # features = np.vstack(features)
                # features_audio = torch.tensor(features).float().to(device)

                # 2. 并行提取egemaps特征
                # 准备帧数据用于并行处理
                frame_data_list = [(frame, sr) for frame in frames]

                # 使用多进程并行处理帧
                with Pool(processes=40) as pool:  # 可根据CPU核心数调整进程数
                    features_list = pool.map(process_egemaps_frame, frame_data_list)

                features = np.vstack(features_list)
                features_audio = torch.tensor(features).float().to(device)

                # 3. 检查/修正NaN
                if torch.isnan(features_audio).any():
                    print(f"ERROR: NaN in egemaps特征，已修正")
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)
                frame_level_audio = features_audio

                # 4. 初始化embedding tensor，写入均值
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                processed_audio_tensor[0] = features_audio.mean(dim=0)

            # elif audio_model == 'mel':
            #     # 1. 加载音频
            #     y, sr = librosa.load(audio_path)
            #     win_length = int(0.02 * sr)
            #     hop_length = int(0.02 * sr)
            #     n_mels = 80
            #     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)
            #     features_audio = torch.tensor(mel).float().permute(1,0)
            #
            #     # 2. 检查/修正NaN
            #     if torch.isnan(features_audio).any():
            #         features_audio = torch.nan_to_num(features_audio, nan=0.0)
            #
            #     # 3. 初始化embedding tensor，写入均值
            #     processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
            #     processed_audio_tensor[0] = features_audio.mean(dim=0)

            # 改为使用torchaudio gpu加速
            elif audio_model == 'mel':
                # 1. 加载音频
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)  # 转mono

                # 2. 保持您现有的参数设置
                win_length = int(0.02 * sample_rate)
                hop_length = int(0.02 * sample_rate)
                n_mels = 80

                # 3. 使用 torchaudio 的 MelSpectrogram 并启用 GPU 加速
                mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=win_length,  # 保持与您原来的 win_length 一致
                    win_length=win_length, hop_length=hop_length, n_mels=n_mels)

                # 4. 如果有 GPU 则使用 GPU 计算
                if torch.cuda.is_available():
                    waveform = waveform.cuda()
                    mel_transform = mel_transform.cuda()

                # 5. 提取 Mel 特征
                mel = mel_transform(waveform)
                features_audio = mel.squeeze(0).permute(1, 0).cpu()  # 转换维度并移回 CPU

                # 6. 检查/修正NaN
                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)

                # 7. 关键：设置 frame_level_audio（与其它分支保持一致）
                frame_level_audio = features_audio

                # 8. 初始化embedding tensor，写入均值
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                processed_audio_tensor[0] = features_audio.mean(dim=0)

            elif audio_model == 'whisper':
                try:
                    # 1. 音频加载和预处理
                    print(f"Loading audio: {audio_path}")
                    audio = whisper.load_audio(audio_path)

                    # 音频质量检查和预处理
                    if len(audio) == 0:
                        print(f"WARNING: Empty audio file {row['uid']}, using default audio")
                        audio = np.zeros(16000)  # 1秒静音
                    else:
                        # 音频归一化防止极端值
                        audio_max = np.max(np.abs(audio))
                        if audio_max > 0:
                            audio = audio / audio_max * 0.95  # 归一化到 [-0.95, 0.95]

                    # 2. Whisper 预处理
                    audio = whisper.pad_or_trim(audio)
                    mel = whisper.log_mel_spectrogram(audio)

                    # 3. Mel 频谱图检查和修复
                    if torch.isnan(mel).any() or torch.isinf(mel).any():
                        print(f"WARNING: NaN/Inf in Mel spectrogram for {row['uid']}")
                        mel = torch.nan_to_num(mel, nan=0.0, posinf=1.0, neginf=-1.0)

                    # 4. 确保在正确的设备上并提取特征
                    mel = mel.to(whisper_model.device)

                    # 5. 提取 encoder 特征
                    with torch.no_grad():
                        features_audio = whisper_model.encoder(mel.unsqueeze(0)).squeeze(0).cpu()

                    # 6. 详细特征检查和修复
                    print(f"Whisper features - Shape: {features_audio.shape}, "
                          f"Range: [{features_audio.min():.3f}, {features_audio.max():.3f}], "
                          f"NaN: {torch.isnan(features_audio).sum().item()}, "
                          f"Inf: {torch.isinf(features_audio).sum().item()}")

                    # 7. 多层次 NaN/Inf 处理
                    # 第一层：基础 NaN/Inf 修复
                    if torch.isnan(features_audio).any() or torch.isinf(features_audio).any():
                        print(f"WARNING: NaN/Inf found in whisper features for {row['uid']}")
                        features_audio = torch.nan_to_num(features_audio, nan=0.0, posinf=1e-6, neginf=-1e-6)

                    # 第二层：限制极端值
                    features_audio = torch.clamp(features_audio, min=-1e3, max=1e3)

                    # 第三层：再次检查（防止 clamp 过程中可能产生的问题）
                    if torch.isnan(features_audio).any() or torch.isinf(features_audio).any():
                        print(f"WARNING: Still have NaN/Inf after clamping for {row['uid']}")
                        features_audio = torch.zeros_like(features_audio)

                    # 8. 特征维度验证
                    if features_audio.numel() == 0:
                        print(f"ERROR: Empty features for {row['uid']}, using fallback")
                        features_audio = torch.zeros((1500, 384))  # Whisper small 模型典型输出

                    # 9. 特别处理均值特征计算
                    mean_features = features_audio.mean(dim=0)
                    if torch.isnan(mean_features).any() or torch.isinf(mean_features).any():
                        print(f"WARNING: NaN/Inf in mean features for {row['uid']}")
                        mean_features = torch.zeros_like(mean_features)

                    # 10. 设置 frame_level_audio（与其它分支保持一致）
                    frame_level_audio = features_audio

                    # 11. 初始化 embedding tensor，写入均值
                    processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                    processed_audio_tensor[0] = mean_features

                    print(f"Successfully processed whisper features for {row['uid']}")

                except Exception as e:
                    print(f"ERROR processing whisper features for {row['uid']}: {str(e)}")
                    # 出错时使用默认特征
                    try:
                        dummy_features = torch.zeros((1500, 384))  # Whisper small 模型输出维度
                        frame_level_audio = dummy_features
                        processed_audio_tensor = torch.zeros((max_length, dummy_features.shape[1]))
                        processed_audio_tensor[0] = dummy_features.mean(dim=0)
                        print(f"Used fallback features for {row['uid']}")
                    except Exception as fallback_error:
                        print(f"FATAL ERROR: Cannot create fallback features for {row['uid']}: {str(fallback_error)}")
                        return -1

            elif audio_model == 'hubert':
                # 1. 加载音频并重采样到16kHz (HuBERT要求16kHz)
                wave_form, sample_rate = torchaudio.load(audio_path)
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)  # 转mono
                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                # 2. 提取HuBERT特征
                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = hubert_model(**inputs_audio)

                last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()

                # 3. 检查/修正NaN
                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)

                # 4. 统一命名和初始化
                features_audio = last_hidden_states_audio
                frame_level_audio = features_audio
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                processed_audio_tensor[0] = features_audio.mean(dim=0)

            elif audio_model == 'wavbert':
                waveform, sample_rate = torchaudio.load(audio_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)  # 转 mono
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
                chunks, chunk_intervals = split_audio_with_fixed_duration(waveform.squeeze(0), sample_rate=16000, max_chunk_duration=150)                # 对每个片段提取特征，并拼接结果
                all_features = []
                for chunk in chunks:
                    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = w2vbert_model(**inputs)
                    features = outputs.last_hidden_state.squeeze(0)
                    all_features.append(features)
                # 直接拼接特征
                last_hidden_states_audio = torch.cat(all_features, dim=0)
                if torch.isnan(last_hidden_states_audio).any():
                    last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)

                features_audio = last_hidden_states_audio
                frame_level_audio = features_audio
                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                processed_audio_tensor[0] = features_audio.mean(dim=0)

            else:
                raise ValueError(f"non support audio_model: {audio_model}")


            # Tokenize and prepare inputs
            inputs_offset = tokenizer(
                transcription,
                return_tensors="pt",
                return_offsets_mapping=True,  # Get token-to-offset mappings
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)


            # Extract word-to-token mapping
                # print(text)
            # 1️⃣ 文本 token 到单词的映射
            input_ids = inputs_offset["input_ids"][-1]
            offset_mapping = inputs_offset["offset_mapping"][-1]

            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            word_mapping = []

            current_word = ""
            current_tokens = []
            current_token_ids = []

            for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
                start, end = offset

                # Skip special tokens ([CLS], [SEP], [PAD])
                if start == 0 and end == 0:
                    continue

                # Check for subwords (##) and group tokens into words
                if token.startswith("##"):
                    current_word += token[2:]
                    current_tokens.append(token)
                    current_token_ids.append(token_id)
                else:
                    # Save previous word
                    if current_word:
                        word_mapping.append((current_word, current_tokens, current_token_ids))
                    # Start a new word
                    current_word = token
                    current_tokens = [token]
                    current_token_ids = [token_id]

            # Save the last word
            if current_word:
                word_mapping.append((current_word, current_tokens, current_token_ids))

            word_level_timestamp_path = os.path.join(root_text_path, row['diagno'], row['uid'] + '.csv')

            # 2️⃣ 读取词级时间戳
            # Read the word level timestamps
            df_word_level = pd.read_csv(word_level_timestamp_path)
            # Columns pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])
            words = []
            for index, data in df_word_level.iterrows():
                words.append((data['word'], data['start'], data['end']))

            idx_probs = 0
            act_word = ''

            idx_att = 0
            idx_start_att = 0

            idx_start_map = 0
            idx_map = 0

            n_audio_segments = 0

            # 3️⃣ 文本 token 与音频时间片段对齐
            # Print results
            for word, tokens, token_ids in word_mapping:
                # print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                cleaned_word = word.replace('Ġ', '')
                act_word += cleaned_word.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()

                # print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                # print(f"Act Word: {act_word}")
                # if idx_probs < len(words):
                    # Check if words[idx_probs][0] is a string before printing
                    # if isinstance(words[idx_probs][0], str):
                    #     print(f"Expected Word: {words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()}")

                # ✅ 情况一：遇到标点符号
                if word.strip() in ['.', ',', '?', '!', ';', 'Ġ','Ġ.', 'Ġ,', 'Ġ?', 'Ġ!', 'Ġ;', 'Ġ...', '...']:    # Ensure only real punctuation
                    if idx_probs > 0:  # Avoid index error
                        start = words[idx_probs-1][2]  # Get last word's end time
                    else:
                        start = 0  # Default to 0 if first word
                    end = words[idx_probs][1] if idx_probs < len(words) else None  # Safe check

                    # 4️⃣ 音频特征切片 + 池化
                    start_segment = math.floor(start * segment_length)
                    end_segment = math.ceil(end * segment_length if end is not None else frame_level_audio.shape[0])
                    # print(f"FOUND PUNCTUATION: {word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                    # print("Token IDs:")
                    # for idx in range(idx_start_map, idx_map + 1):
                    #     print(f"{idx}: {word_mapping[idx]}")
                    #     print('------------------------------------------')

                    for idx in range(idx_start_att, idx_att + len(token_ids)):
                        n_audio_segments += 1

                        if end_segment - start_segment < 3:
                            start_segment = max(0, start_segment - 2)
                            end_segment = min(frame_level_audio.shape[0], end_segment + 2)

                        audio_features_segment = frame_level_audio[start_segment:end_segment]
                        processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                    idx_start_att = idx_att + len(token_ids)
                    idx_start_map = idx_map + 1


                # ✅ 情况二：遇到普通词
                if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word == words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                    start = words[idx_probs][1]
                    end = words[idx_probs][2]

                    # 4️⃣ 音频特征切片 + 池化
                    start_segment = math.floor(start * segment_length)
                    end_segment = math.ceil(end * segment_length if end is not None else frame_level_audio.shape[0])

                    # print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                    # print("Token IDs:")
                    # for idx in range(idx_start_map, idx_map + 1):
                    #     print(f"{idx}: {word_mapping[idx]}")
                    #     print('------------------------------------------')

                    for idx in range(idx_start_att, idx_att + len(token_ids)):
                        n_audio_segments += 1

                        if end_segment - start_segment < 3:
                            start_segment = max(0, start_segment - 2)
                            end_segment = min(frame_level_audio.shape[0], end_segment + 2)


                        audio_features_segment = frame_level_audio[start_segment:end_segment]
                        processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                    idx_probs += 1
                    act_word = ''
                    idx_start_att = idx_att + len(token_ids)
                    idx_start_map = idx_map + 1

                idx_att += len(token_ids)
                idx_map += 1

            if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word in words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                start = words[idx_probs][1]
                end = words[idx_probs][2]

                start_segment = math.floor(start * segment_length)
                end_segment = math.ceil(end * segment_length if end is not None else frame_level_audio.shape[0])

                # print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                # print("Token IDs:")
                # for idx in range(idx_start_map, idx_map):
                #     print(f"{idx}: {word_mapping[idx]}")
                #     print('------------------------------------------')

                for idx in range(idx_start_att, idx_att):
                    n_audio_segments += 1

                    if end_segment - start_segment < 3:
                        start_segment = max(0, start_segment - 2)
                        end_segment = min(frame_level_audio.shape[0], end_segment + 2)


                    audio_features_segment = frame_level_audio[start_segment:end_segment]
                    processed_audio_tensor[idx + 1] = torch.clamp(audio_features_segment.mean(dim=0), min=-1e3, max=1e3)

                idx_probs += 1
                act_word = ''
                idx_start_att = idx_att + len(token_ids)
                idx_start_map = idx_map + 1


            # print(f"Number of audio segments: {n_audio_segments}")
            # See inputs numbers and compare with the number of audio segments, separate with PAD tokens, eclusding them
            #total_tokens = torch.sum(inputs_text['input_ids'][0] != 0).item()
            total_tokens = torch.sum(inputs_text['attention_mask'][0]).item()
            # print(f"Total tokens: {total_tokens}")
            # 5️⃣ 边界检查与异常处理
            if n_audio_segments + 2 != total_tokens:
                print(f"ERROR in {row['diagno']}, {row['uid']}: Number of audio segments ({n_audio_segments}) does not match the number of tokens ({total_tokens})")
                print(f"Completed audios: {completed_audios}")
                return -1

            if torch.isnan(processed_audio_tensor).any():
                processed_audio_tensor = torch.nan_to_num(processed_audio_tensor, nan=0.0)
                print(f"ERROR in {row['diagno']}, {row['uid']}: NaN values in processed_audio_tensor, and padded with 0")
                print(f"Completed audios: {completed_audios}")
                # return -1

            # 用 os.path.splitext 去掉扩展名，再拼接 _audio.pt
            audio_uid = os.path.splitext(row['uid'])[0]
            torch.save(processed_audio_tensor.cpu(), os.path.join(root_text_path, row['diagno'], audio_uid + textual_model_data + pauses_data + audio_model_data + '.pt'))
            # 文本嵌入和声学嵌入在该代码中是通过时间对齐和tokenization策略绑定在一起的，因为不同文本模型的tokenization策略和输出维度可能不同


            completed_audios += 1
        # ========== 循环末尾清理 ==========
            del processed_audio_tensor
            if 'frame_level_audio' in locals(): del frame_level_audio
            if 'features_audio' in locals(): del features_audio
            if 'inputs_audio' in locals(): del inputs_audio
            if 'outputs_audio' in locals(): del outputs_audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
         # ========== 清理文本变量 ==========
        del inputs_text, outputs_text, last_hidden_states_text
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"------------------------------------------")
        print(f"CORRECTLY PROCESSED ALL AUDIOS")
        print(f"Completed audios: {completed_audios}")



if __name__ == '__main__':
    preprocess_text()
    print(f'textual_model = {textual_model}, audio_model = {audio_model}, pause = {pauses}')
