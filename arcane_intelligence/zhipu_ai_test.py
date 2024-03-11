from resource.zhipu_ai import ZhipuAIProvider
from resource.schema import ChatMessage
import os
import asyncio
import logging

async def main():
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

  ai_provider = ZhipuAIProvider(os.environ.get("ZHIPU_API_KEY"))

  chat_history = []

  system_message = ChatMessage.system("你是人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。")
  user_message = ChatMessage.user("你好，我是羊肉，2 + 2等于多少")

  chat_history.append(system_message)
  chat_history.append(user_message)

  response = await ai_provider.create_chat_completion(chat_history, model_name="glm-4")

  chat_history.append(response.result)

  logging.info("{}".format(response))

  user_message2 = ChatMessage.user("你好，你记得我刚才说了什么吗")
  chat_history.append(user_message2)

  response = await ai_provider.create_chat_completion(messages=chat_history, model_name="glm-4")

  logging.info("{}".format(response))

if __name__ == "__main__":
  asyncio.run(main())