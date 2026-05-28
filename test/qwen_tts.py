import dashscope
import os

API_KEY = "sk-5ef1b86756e34a9089dc28c072fe6e7c"  # 替换为你的阿里云百炼 API Key

# 以下为北京地域url，若使用新加坡地域的模型，需将url替换为：https://dashscope-intl.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'


text = "今天心情不错，出去稍微走走，买杯咖啡，然后就坐在路边看看人来人往的。"
# SpeechSynthesizer接口使用方法：dashscope.audio.qwen_tts.SpeechSynthesizer.call(...)
response = dashscope.MultiModalConversation.call(
    # 如需使用指令控制功能，请将model替换为qwen3-tts-instruct-flash
    model="qwen3-tts-flash",
    api_key=API_KEY,
    text=text,
    voice="Bunny",
    language_type="Chinese", # 建议与文本语种一致，以获得正确的发音和自然的语调。
    # 如需使用指令控制功能，请取消下方注释，并将model替换为qwen3-tts-instruct-flash
    # instructions='语速较快，带有明显的上扬语调，适合介绍时尚产品。',
    # optimize_instructions=True,
    stream=False

)
print(response)