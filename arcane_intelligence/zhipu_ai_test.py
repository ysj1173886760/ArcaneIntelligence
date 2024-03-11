from resource.zhipu_ai import ZhipuAIProvider
from resource.schema import ChatMessage
from resource.schema import CompletionModelFunction
from utils import JSONSchema
import os
import asyncio
import logging
import math

def cosine_distance(A, B):
    # 计算点积
    dot_product = sum(a * b for a, b in zip(A, B))
    
    # 计算A和B的欧几里得范数
    norm_A = math.sqrt(sum(a**2 for a in A))
    norm_B = math.sqrt(sum(b**2 for b in B))
    
    # 计算余弦相似度
    cosine_similarity = dot_product / (norm_A * norm_B)
    
    # 计算余弦距离
    cosine_distance = 1 - cosine_similarity
    
    return cosine_distance

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

async def embedding_test():
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

  ai_provider = ZhipuAIProvider(os.environ.get("ZHIPU_API_KEY"))

  model_name = "embedding-2"

  response = await ai_provider.create_embedding("羊头", model_name=model_name)
  # logging.info("{}".format(response))
  sheep_head_vector = response.embedding

  response = await ai_provider.create_embedding("羊肉", model_name=model_name)
  # logging.info("{}".format(response))
  sheep_meat_vector = response.embedding

  response = await ai_provider.create_embedding("猪头", model_name=model_name)
  # logging.info("{}".format(response))
  pig_head_vector = response.embedding

  response = await ai_provider.create_embedding("飞机", model_name=model_name)
  plane_vector = response.embedding

  response = await ai_provider.create_embedding("Represent the question for retrieving supporting documents: 飞机", model_name=model_name)
  test_vector = response.embedding

  logging.info("{}".format(cosine_distance(sheep_head_vector, sheep_meat_vector)))
  logging.info("{}".format(cosine_distance(sheep_head_vector, pig_head_vector)))
  logging.info("{}".format(cosine_distance(sheep_head_vector, plane_vector)))
  logging.info("{}".format(cosine_distance(plane_vector, test_vector)))

search_with_keywords_func = CompletionModelFunction(
  name="search_with_keywords",
  description=(
    "Search the document with the keywords"
  ),
  parameters={
    "key_words": JSONSchema(
      type=JSONSchema.Type.ARRAY,
      items=JSONSchema(
        type=JSONSchema.Type.STRING
      ),
      minItems=1,
      required=True
    )
  }
)

async def function_test():
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
  ai_provider = ZhipuAIProvider(os.environ.get("ZHIPU_API_KEY"))

  chat_history = []
  system_message = ChatMessage.system("你是人工智能助手，负责帮助用户辅助搜索文档")
  user_message = ChatMessage.user("请帮我搜索有关戴森球的科幻小说")
  chat_history.append(system_message)
  chat_history.append(user_message)

  kwargs = {}
  kwargs["tools"] = [
    {"type": "function", "function": search_with_keywords_func.schema}
  ]
  kwargs["tool_choice"] = {
    "type": "function",
    "function": {"name": search_with_keywords_func.name}
  }

  response = await ai_provider.create_chat_completion(chat_history, "glm-4", **kwargs)

  logging.info(response)

if __name__ == "__main__":
  # asyncio.run(main())
  # asyncio.run(embedding_test())
  asyncio.run(function_test())
  