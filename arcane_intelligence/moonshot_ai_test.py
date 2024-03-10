from resource.moonshot_ai import MoonshotAIProvider
import os
import asyncio
import logging

async def main():
  logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

  ai_provider = MoonshotAIProvider(api_key=os.environ.get("MOONSHOT_API_KEY"))

  user_message = "你好，我是羊肉，2 + 2等于多少"

  response = await ai_provider.create_embedding(user_message, "moonshot-v1-8k")

  logging.info("{}".format(response))

if __name__ == "__main__":
  asyncio.run(main())