from zhipuai import ZhipuAI
from zhipuai.types.chat.chat_completion import CompletionMessage
from zhipuai.types.embeddings import EmbeddingsResponded
from resource.schema import ChatMessage, ModelResponse, ChatModelProvider, AssistantChatMessage, EmbeddingModelResponse, EmbeddingModelProvider, ChatModelResponse
import logging
from typing import List

class ZhipuAIProvider(ChatModelProvider, EmbeddingModelProvider):
  _client: ZhipuAI

  def __init__(self, api_key: str):
    self._client = ZhipuAI(api_key=api_key)

  @staticmethod
  def _parse_completion_message(chat_completion_message: CompletionMessage) -> AssistantChatMessage:
      return AssistantChatMessage(role=chat_completion_message.role, content=chat_completion_message.content)

  async def create_chat_completion(self, messages: list[ChatMessage], model_name: str, **kwargs) -> ChatModelResponse:
    raw_messages = [
      message.dict(include={"role", "content"}) for message in messages
    ]
    chat_completion = self._client.chat.completions.create(messages=raw_messages, model=model_name)

    logging.debug('chat response: {} prompt token used: {} completion token used: {} total token used: {}'.format(chat_completion.choices[0].message.content, chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens, chat_completion.usage.total_tokens))

    return ChatModelResponse(result=ZhipuAIProvider._parse_completion_message(chat_completion.choices[0].message), prompt_tokens_used=chat_completion.usage.prompt_tokens, completion_tokens_used=chat_completion.usage.completion_tokens)

  @staticmethod
  def _parse_embedding(embedding_response: EmbeddingsResponded) -> List[float]:
      return embedding_response.data[0].embedding
  
  async def create_embedding(self, text: str, model_name: str) -> EmbeddingModelResponse:
    embedding_response : EmbeddingsResponded = self._client.embeddings.create(input=text, model=model_name)

    logging.debug('create embedding response: {} prompt token used: {} completion token used: {} total token used: {}'.format(embedding_response.data, embedding_response.usage.prompt_tokens, embedding_response.usage.completion_tokens, embedding_response.usage.total_tokens))
    
    return EmbeddingModelResponse(embedding=ZhipuAIProvider._parse_embedding(embedding_response), prompt_tokens_used=embedding_response.usage.prompt_tokens, completion_tokens_used=embedding_response.usage.completion_tokens)