# an implementation of tau bench in openbench 

"""
Basic flow is as follows:
1. get user vars. We need to read the env type, the agent strat, and model names for both agent and user simulation
2. we run the data loader. The dataloader will rely on the env to load the corresponding tasks
3. we run the agent and user simulation. This will be handled in the custom solver
4. the solver returns the full trace of the simulation
5. we run the scorer. The scorer will follow the pass^k logic from the paper

A lot of moving parts, but tasks are prebuilt so a lot of the logic is in execution which makes testing hard



"""


from inspect_ai import Task,task
from openbench.solvers.taubench.taubench_solver import taubench_solver
from .environments.env import get_task_env
# import data laoding emthods
from openbench.datasets.taubench.taubench_load_data import get_taubench_dataset,get_task_env_data



# lets hard set some varaibles for now


@task
def tau_bench(env_type: str = "retail", agent_model: str = 'gpt-4o-mini', agent_strat: str = 'llm', user_model: str = 'gpt-4o-mini', user_strat: str = 'llm',task_split: str = "test")-> Task:
    # default seed
    print("Using seed: ",10)
    seed=10

    # the flow is to make our simulated user env, our agent, and then in the solver
    # loop the agent execution pathway which calls tools and the step function in the user env
    # agent should have internal access to copy of task_env_data. it should retain changes
    # the env has the scoring functions
    # until the task is done. We return ALL steps. 

    dataset = get_taubench_dataset(env_type,task_split,metadata={
            "env_type": env_type,
            "user_strat": user_strat,
            "user_model": user_model,
            "agent_model": agent_model,
            "task_split": task_split,
            "agent_strat": agent_strat
        })
    
    return Task(
        dataset=dataset,
        solver=[
         taubench_solver()   
        ],
    )