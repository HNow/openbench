import copy

from inspect_ai.solver import solver, TaskState, Generate
from openbench.evals.taubench.environments.env import get_task_env
from typing import Callable
from openbench.evals.taubench.agents.simulated_user_agent import LLMUserSimulationEnv
from openbench.evals.taubench.agents.tool_call_agent import tool_calling_agent
from inspect_ai.agent import AgentState


@solver
def taubench_solver() -> Callable:

    # with the task environment lets initialize both the simulated user agent and the task agent
    async def solve(state: TaskState,generate: Generate) -> TaskState:
        # unpack the state metadata
        print(state.metadata,"\n\n\n STATE METADATA")

        env_type = state.metadata["env_type"]
        user_strat = state.metadata["user_strat"]
        user_model = state.metadata["user_model"]
        agent_model = state.metadata["agent_model"]
        task_split = state.metadata["task_split"]
        agent_strat = state.metadata["agent_strat"] 

        task_env = get_task_env(env_type,user_strat,user_model,agent_model,task_split)
        print(task_env,"\n\n\n TASK ENV")
        # now I make my agent and simulated user agent
        if(user_strat == "llm"):
            simulated_user=LLMUserSimulationEnv(task_env.user_model_provider,task_env.user_model)
        else:
            raise ValueError(f"Unknown user strategy: {user_strat}")

        # now we can make the task agent and have it carry out its call loop

        # run the task agent
        # TODO, make this respective of type of agent
        task_agent = tool_calling_agent(task_env.agent_model_provider,task_env.agent_model,task_env.wiki,task_env.tools)

        # run the task agent
        working_env_data=copy.deepcopy(task_env.data)
        messages = await task_agent(AgentState(messages=[]),simulated_user,state.metadata,working_env_data)

        return state


    # now we can make the task agent and have it carry out its call loop

    return solve
