from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types import CreateEmbeddingResponse
from openai.types.embedding import Embedding
from resource.schema import ChatMessage, ModelResponse, ChatModelProvider, AssistantChatMessage, EmbeddingModelResponse, EmbeddingModelProvider
import logging

class MoonshotAIProvider(ChatModelProvider, EmbeddingModelProvider):
  _api_key: str = ""
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
  
  async def create_embedding(self, text: str, model_name: str) -> EmbeddingModelResponse:
    embedding_response : CreateEmbeddingResponse = await self._client.embeddings.create(input=text, model=model_name)

    logging.debug('create embedding response: {} prompt token used: {} total token used: {}'.format(embedding_response.data, embedding_response.usage.prompt_tokens, embedding_response.usage.total_tokens))
    
    return EmbeddingModelResponse(embedding=embedding_response.data.embedding, prompt_tokens_used=embedding_response.usage.prompt_tokens, completion_tokens_used=embedding_response.usage.total_tokens)
    