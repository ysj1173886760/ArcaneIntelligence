import abc
from typing import Callable, Literal, Optional
from pydantic import BaseModel
import enum

class ChatMessage(BaseModel):
    class Role(str, enum.Enum):
        USER = "user"
        SYSTEM = "system"
        ASSISTANT = "assistant"

    role: Role
    content: str

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.USER, content=content)

    @staticmethod
    def system(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.SYSTEM, content=content)

class AssistantChatMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str]

class ModelResponse(BaseModel):
    """Standard response struct for a response from a model."""

    prompt_tokens_used: int
    completion_tokens_used: int
    result: AssistantChatMessage
    # model_info: ModelInfo

class ChatModelProvider():
  @abc.abstractmethod
  async def create_chat_completion(
      self,
      messages: list[ChatMessage],
      model_name: str,
      **kwargs,
  ) -> ModelResponse:
      ...

