from typing import Optional, List
from inspect_ai.model import (
    get_model,
    Model,
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    GenerateConfig 
)

class LLMUserSimulationEnv:
    """
    Stateless wrapper over Inspect AI.
    - Holds a Model.
    - Does NOT store transcript; caller owns it.
    - Methods return updated message lists.
    """

    def __init__(self, model_name: str, model_provider: str) -> None:
        self.model_name = model_name
        self.model_provider = model_provider
        self.model = get_model(model_name=model_provider+"/"+model_name,config=GenerateConfig(temperature=0.0))

    @staticmethod
    def build_system_prompt(instruction: Optional[str]) -> str:
        instr = f"\n\nInstruction: {instruction}\n" if instruction else ""
        return (
            "You are a user interacting with an agent."
            f"{instr}"
            "Rules:\n"
            "- Just generate one line at a time to simulate the user's message.\n"
            "- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n"
            "- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n"
            "- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n"
            "- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n"
            "- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
        )

    # call this first as its from the reset() flow from tau bench repo
    def seed_transcript(
        self,
        instruction: Optional[str] = None,
        opening_user: Optional[str] = "Hi! How can I help you today?",
    ) -> List[ChatMessage]:
        msgs: List[ChatMessage] = [ChatMessageSystem(content=self.build_system_prompt(instruction))]
        if opening_user:
            msgs.append(ChatMessageUser(content=opening_user))
        print(msgs,"\n\n\n SEED TRANSCRIPT")
        return msgs

    async def generate_next_message(
        self,
        transcript: List[ChatMessage],
    ) -> List[ChatMessage]:
        out = await self.model.generate(transcript)
        if not isinstance(out.message, ChatMessageAssistant):
            raise TypeError("ModelOutput.message must be ChatMessageAssistant")
        return [*transcript, out.message]

    async def step(
        self,
        transcript: List[ChatMessage],
        user_content: str,
    ) -> List[ChatMessage]:
        with_user = [*transcript, ChatMessageUser(content=user_content)]
        return await self.generate_next_message(with_user)
