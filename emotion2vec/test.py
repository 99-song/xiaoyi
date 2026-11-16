# from funasr import AutoModel


# model_path = "/home/songhongyi/data/xiaoyi/model/emotion2vec_base_finetuned"
# model = AutoModel(
#     model=model_path,
#     hub="ms",
# )

# wav_file = "/home/songhongyi/data/xiaoyi/backend/db_data/wavs/6ea27ca43709452ebb1db3b938a3be8e/assistant_15db069b4d55428e98db94811b8f4f0e.wav"
# rec_result = model.generate(
#     wav_file,
#     output_dir="./outputs",
#     granularity="utterance",
#     extract_embedding=False,
# )

# # 提取情感标签
# emotion_dict = {}
# if rec_result and isinstance(rec_result, list):
#     item = rec_result[0]                     # 取第一个结果（utterance级别只有一个）
#     scores = item["scores"]
#     labels = item["labels"]
#     max_index = scores.index(max(scores))    # 找到最高分索引
#     best_label = labels[max_index]           # 对应的标签字符串，例如 "开心/happy"
#     # 如果只需要中文，可以切分一下
#     chinese_label = best_label.split("/")[0] # "开心"
#     emotion_dict = {"emotion": chinese_label}
#     # 或者保留原始 label: emotion_dict = {"emotion": best_label}

# print(emotion_dict)   # 输出: {'emotion': '开心'}


from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "/home/songhongyi/data/xiaoyi/model/SenseVoiceSmall"


model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",  
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    ban_emo_unk=True
)

# en
res = model.generate(
    input=f"/home/songhongyi/data/xiaoyi/backend/db_data/wavs/1c4a98d4cce543f692968a98b1ea924a/assistant_6581b583b17b48cb9304bcc0e2218612.wav",
    cache={},
    language="zn",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  
    merge_length_s=15,

)
print(res)
text = rich_transcription_postprocess(res[0]["text"])
print(text)
