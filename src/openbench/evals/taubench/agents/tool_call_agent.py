import json
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import (
    get_model,
    Model,
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    ChatMessageAssistant,
    GenerateConfig ,
    ModelOutput
)
from openbench.evals.taubench.agents.simulated_user_agent import LLMUserSimulationEnv
from typing import List


RESPOND_ACTION_NAME = "respond"

@agent
def tool_calling_agent(agent_model: str, agent_model_provider: str,wiki: str,tools: List) -> Agent:

    # initially we wat to use the user_simulation_env to seed the chat history
    async def run(state: AgentState,user_simulation_env: LLMUserSimulationEnv,task_metadata: dict,task_env_data: dict) -> AgentState:
        model = get_model(model_name=agent_model_provider+"/"+agent_model,config=GenerateConfig(temperature=0.0,system_message=wiki))

        # list of dicts to store tool calls and params
        tool_calls = []

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
        print(response,"\n\n\n initial response")


        # now we initialize our AgentState messages
        new_messages = [
            ChatMessageSystem(content=wiki),
            ChatMessageUser(content=initial_query),
        ]

        # overwrite (uses your property setter -> wraps in ChatMessageList)
        state.messages = new_messages
        
        for _ in range(10):
            # i need the tools now, these should be fully defined in the env i think
            response = await model.generate(state.messages,tools=tools)
            # response = await model.generate("what is the sum of 25 and 10?",tools=tools)
            # with this response we can now look over it for any tool calls and other content

            print(response,"\n\n\n RESPONSE")
            model_dump=response.choices[0]
            # add the ChatMessage to the state.messages
            state.messages.append(model_dump.message)
            # now with the model dump we can move to the tool call processing
            if hasattr(model_dump, 'tool_calls') and model_dump.tool_calls is not None:
                # extract the function name and params
                function_name = model_dump.tool_calls[0].function
                params = json.loads(model_dump.tool_calls[0].arguments)
                # lets put these into the tool calls dict
                tool_calls.append({"function_name": function_name, "params": params})
            else:
                # if there are no tool calls, just add the response message
                tool_calls.append({"function_name": RESPOND_ACTION_NAME, "params": {"content": model_dump.message.content}})


            # now we simulate env stepping
            current_reward=0
            tool_result=None
            done=False
            if tool_calls[-1]["function_name"] == RESPOND_ACTION_NAME:
                # print state prior
                print(state.messages,"\n\n\n STATE PRIOR")
                response=await user_simulation_env.step(state.messages,tool_calls[-1]["params"]["content"])
                # function returns the whole updated list so lets just set the whole state to that
                state.messages = response
                # print state after
                print(state.messages,"\n\n\n STATE AFTER")
                done = "###STOP###" in state.messages[-1].content
            # tool map should be made in the tool init, just a map of names to functions
            # tau bench uses their Activity data type, model may output an unknown tool name though so better to check
            elif tool_calls[-1]["function_name"] in tools:
                # now we can invoke the tool
                # need to determine how inspect ai wants me to do this
                # should be try catch probably and also need to pass in env data
                tool_result = tools[tool_calls[-1]["function_name"]](**tool_calls[-1]["params"])
                # add the tool result to the state messages
            else:
                # unknown action
                tool_result = "Unknown action: " + tool_calls[-1]["function_name"]
            # done env stepping
            # we should now have all of the parts to build the next user agent reponse and then loop
            
             

        # update and return state
        state.output = response.message.content
        state.messages.extend(messages)
        return state

    return run