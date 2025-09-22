from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import (
    get_model,
    Model,
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    GenerateConfig 
)
from openbench.evals.taubench.agents.simulated_user_agent import LLMUserSimulationEnv
from typing import List




@agent
def tool_calling_agent(agent_model: str, agent_model_provider: str,wiki: str) -> Agent:

    # initially we wat to use the user_simulation_env to seed the chat history
    async def run(state: AgentState,user_simulation_env: LLMUserSimulationEnv,task_metadata: dict) -> AgentState:
        model = get_model(model_name=agent_model_provider+"/"+agent_model,config=GenerateConfig(temperature=0.0,system_message=wiki))

        # we now have our empty messages list


        # run a tool loop w/ the LLMUserSimulationEnv

        # effectively we put our version of the 'solve'loop here fromt he agent declarations in tau bench

        initial_instruction = task_metadata["instruction"]
        # now we implement the env_reset_res call on line 31 of tool calling agent in tau bench
        chatMessages = user_simulation_env.seed_transcript(initial_instruction)
        # parse out the actual response text
        response = await user_simulation_env.generate_next_message(transcript=chatMessages)
        # update response to only be actual content
        initial_query = response[-1].content
        # print it for testing
        print(response,"\n\n\n RESPONSE")


        # now we initialize our AgentState messages
        new_messages = [
            ChatMessageSystem(content=wiki),
            ChatMessageUser(content=initial_query),
        ]

        # overwrite (uses your property setter -> wraps in ChatMessageList)
        state.messages = new_messages
        
        for _ in range(10):
            # i need the tools now, these should be fully defined in the env i think
            response = await model.generate(state.messages,tools=[])
            # parse out the actual response from the fully returned ChatMessages list
            

        

        # update and return state
        state.output = response.message.content
        state.messages.extend(messages)
        return state

    return run