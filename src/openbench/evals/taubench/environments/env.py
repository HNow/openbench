# define the base environment class 

from openbench.datasets.taubench.taubench_load_data import get_retail_env_data, get_airline_env_data
from typing import Callable, Dict, Any, List
from openbench.datasets.taubench.data.retail.wiki import retail_wiki
from openbench.datasets.taubench.data.retail.rules import retail_rules
from openbench.datasets.taubench.data.retail.tools.init_tools import retail_tools
from inspect_ai.tool import Tool


class Env(object):
    def __init__(self, data_load_func: Callable[[], Dict[str, Any]], wiki: str, rules: List[str], user_strat: str, user_model: str, user_model_provider: str, agent_model: str, agent_model_provider: str, tools: List[Tool]) -> None:
        super().__init__()
        self.data_load_func = data_load_func
        self.data = self.data_load_func()
        self.wiki = wiki
        self.rules = rules
        self.tools = tools
        self.user_strat = user_strat
        self.user_model = user_model
        self.user_model_provider = user_model_provider
        self.agent_model = agent_model
        self.agent_model_provider = agent_model_provider


def get_task_env(env_type: str, user_strat: str, user_model: str, agent_model: str, task_split: str) -> Env:
    if env_type == "retail":
        data_load_func = lambda: get_retail_env_data()
        wiki = retail_wiki
        rules = retail_rules
        tools = retail_tools
    elif env_type == "airline":
        data_load_func = lambda: get_airline_env_data()
        wiki = "airline_wiki"   
        rules = ["airline_rules"]
        tools = []
    else:
        raise ValueError(f"Unsupported environment type: {env_type}")
    # TODO: un-hardcode this after testing
    return Env(
        data_load_func=data_load_func,
        wiki=wiki,
        rules=rules,
        tools=tools,
        user_strat=user_strat,
        user_model=user_model,
        user_model_provider="openai",
        agent_model=agent_model,
        agent_model_provider="openai"
    )
