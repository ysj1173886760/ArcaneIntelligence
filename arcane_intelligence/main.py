from resource.moonshot_ai import MoonshotAIProvider
from resource.schema import ChatMessage
from configuration import GlobalConfig
import os
import asyncio
import logging


async def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python config.py <config_file_path>")
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    config_file_path = sys.argv[1]
    GlobalConfig.init_config(config_file_path)

    ai_provider = MoonshotAIProvider(os.environ.get("MOONSHOT_API_KEY"))

    chat_history = []

    system_message = ChatMessage.system(
        "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。"
    )
    user_message = ChatMessage.user("你好，我是羊肉，2 + 2等于多少")

    chat_history.append(system_message)
    chat_history.append(user_message)

    response = await ai_provider.create_chat_completion(
        [system_message, user_message], "moonshot-v1-8k"
    )

    chat_history.append(response.result)

    logging.info("{}".format(response))

    user_message2 = ChatMessage.user("你好，你记得我刚才说了什么吗")
    chat_history.append(user_message2)

    response = await ai_provider.create_chat_completion(chat_history, "moonshot-v1-8k")

    logging.info("{}".format(response))


if __name__ == "__main__":
    asyncio.run(main())
