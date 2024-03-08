import openai

# 设置你的API密钥
openai.api_key = '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'

# 调用GPT-4 API，假设你已经有了一个图片的URL或文件ID
response = openai.Image.create(
  file=open("/share/home/dq070/hy-tmp/AS_id_all/all-condition/sd_rfs_twilight/GOPR0122_frame_000161_rgb_anon.png", "rb"),
  model="gpt4",  # 选择合适的模型
  prompt="Generate a text prompt based on this image."
)

# 打印生成的文本提示
print(response['choices'][0]['text'])
