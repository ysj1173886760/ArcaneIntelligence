import abc
from typing import Callable, Literal, Optional, TypedDict
from pydantic import BaseModel, Field
from utils.json_schema import JSONSchema
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

class ChatMessageDict(TypedDict):
    role: str
    content: str

class AssistantChatMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str]

class ModelResponse(BaseModel):
    """Standard response struct for a response from a model."""

    prompt_tokens_used: int
    completion_tokens_used: int
    result: AssistantChatMessage
    # model_info: ModelInfo

Embedding = list[float]

class EmbeddingModelResponse(ModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: Embedding = Field(default_factory=list)

class EmbeddingModelProvider(abc.ABC):
  @abc.abstractmethod
  async def create_embedding(self, text: str, model_name: str) -> EmbeddingModelResponse:
    ...

class ChatModelProvider(abc.ABC):
  @abc.abstractmethod
  async def create_chat_completion(
      self,
      messages: list[ChatMessage],
      model_name: str,
      **kwargs,
  ) -> ModelResponse:
      ...

class CompletionModelFunction(BaseModel):
    """General representation object for LLM-callable functions."""

    name: str
    description: str
    parameters: dict[str, "JSONSchema"]

    @property
    def schema(self) -> dict[str, str | dict | list]:
        """Returns an OpenAI-consumable function specification"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: param.to_dict() for name, param in self.parameters.items()
                },
                "required": [
                    name for name, param in self.parameters.items() if param.required
                ],
            },
        }

    @staticmethod
    def parse(schema: dict) -> "CompletionModelFunction":
        return CompletionModelFunction(
            name=schema["name"],
            description=schema["description"],
            parameters=JSONSchema.parse_properties(schema["parameters"]),
        )

    def fmt_line(self) -> str:
        params = ", ".join(
            f"{name}{'?' if not p.required else ''}: " f"{p.typescript_type}"
            for name, p in self.parameters.items()
        )
        return f"{self.name}: {self.description}. Params: ({params})"

