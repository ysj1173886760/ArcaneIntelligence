from typing import Any, List, Mapping, Optional
from resource.zhipu_ai import ZhipuAIProvider
from resource.schema import ChatMessage
import asyncio

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

MODEL_NAME = "glm-4"

class ZhipuLLM(LLM):
  _provider: ZhipuAIProvider
  
  def __init__(self, provider: ZhipuAIProvider):
    self._provider = provider
  
  @property
  def _llm_type(self) -> str:
      return "custom"

  def _call(
      self,
      prompt: str,
      stop: Optional[List[str]] = None,
      run_manager: Optional[CallbackManagerForLLMRun] = None,
      **kwargs: Any,
  ) -> str:
      if stop is not None:
          raise ValueError("stop kwargs are not implemented.")
      user_message = ChatMessage.user(prompt)
      response = asyncio.run(self._provider.create_chat_completion(messages=[user_message], model_name=MODEL_NAME))

      if response.result.content is None:
        return ""

      return response.result.content

  @property
  def _identifying_params(self) -> Mapping[str, Any]:
      """Get the identifying parameters."""
      return {"provider": self._provider}