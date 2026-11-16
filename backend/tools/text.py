import re
def clean_text_for_tts(text: str) -> str:
    # 移除常见的 Markdown 符号 (*, #, `, ~, -, >)
    text = re.sub(r'[*#`~>]+', '', text)
    # 移除纯英文破折号等可能导致停顿过长或报错的符号
    text = text.replace('---', '').replace('--', '')
    return text.strip()