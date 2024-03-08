from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from resource.schema import ChatMessage, ModelResponse, ChatModelProvider, AssistantChatMessage
import logging

class MoonshotAIProvider(ChatModelProvider):
  _api_key: str
  _base_url: str = "https://api.moonshot.cn/v1"
  _client: AsyncOpenAI

  def __init__(self, api_key: str):
    self._api_key = api_key
    self._client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)


  @staticmethod
  def _parse_completion_message(chat_completion_message: ChatCompletion) -> AssistantChatMessage:
      return AssistantChatMessage(role=chat_completion_message.choices[0].message.role, content=chat_completion_message.choices[0].message.content)
    
  async def create_chat_completion(self, messages: list[ChatMessage], model_name: str, **kwargs) -> ModelResponse:
    raw_messages = [
      message.dict(include={"role", "content"}) for message in messages
    ]
    chat_completion = await self._client.chat.completions.create(model=model_name, messages=raw_messages, temperature=0.3)

    logging.debug('chat response: {} prompt token used: {} completion token used: {}'.format(chat_completion.choices[0].message.content, chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens))

    return ModelResponse(result=MoonshotAIProvider._parse_completion_message(chat_completion), prompt_tokens_used=chat_completion.usage.prompt_tokens, completion_tokens_used=chat_completion.usage.completion_tokens)
  