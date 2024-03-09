import abc
from pydantic import BaseModel, Field
from resource.schema import ChatMessage, AssistantChatMessage, CompletionModelFunction, ChatMessageDict

class ChatPrompt(BaseModel):
    messages: list[ChatMessage]
    functions: list[CompletionModelFunction] = Field(default_factory=list)

    def raw(self) -> list[ChatMessageDict]:
        return [m.dict() for m in self.messages]

    def __str__(self):
        return "\n\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in self.messages
        )

class PromptStrategy(abc.ABC):
    @property

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessage):
        ...