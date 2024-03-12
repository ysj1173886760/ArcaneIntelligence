import logging

from pydantic import BaseModel
from prompting.schema import PromptStrategy, ChatPrompt
from resource.schema import ChatMessage, AssistantChatMessage


class InitialPlan(PromptStrategy):
    DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
        "You are an expert project planner. "
        "Your responsibility is to create work plans for autonomous agents. "
        "You will be given a name, a role, set of goals for the agent to accomplish. "
        "Your job is to break down those goals into a set of tasks that the agent can"
        " accomplish to achieve those goals. "
        "Agents are resourceful, but require clear instructions."
        " Each task you create should have clearly defined `ready_criteria` that the"
        " agent can check to see if the task is ready to be started."
        " Each task should also have clearly defined `acceptance_criteria` that the"
        " agent can check to evaluate if the task is complete. "
        "You should create as many tasks as you think is necessary to accomplish"
        " the goals.\n\n"
    )

    DEFAULT_USER_PROMPT_TEMPLATE = (
        "You are {agent_name}, {agent_role}\n" "Your goals are:\n" "{agent_goals}"
    )

    def __init__(self):
        ...

    def build_prompt(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        abilities: list[str],
    ) -> ChatPrompt:
        ...

    def parse_response_content(self, response_content: AssistantChatMessage):
        ...
