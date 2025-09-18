"""
A first attempt at implementing LongBench v2
https://longbench2.github.io/


TODO:
1) load data
2) split tasks
2 b) see if tasks can run parallel?


NOTE:
max context length of model should be taken from user on run for removal of datapoints that are too long
benchmark has multiple prompt templates, going with 0shot, look into templating for tasks 
"""


from inspect_ai import Task, task
from inspect_ai.solver import generate
from inspect_ai.model import GenerateConfig
from openbench.datasets.longbench_v2 import get_dataset
from openbench.scorers.longbench_v2 import longbench_v2_scorer





# this is how we define a task
# but what if i need multiple subtasks? is that a bad pattern here?
# lets start simple with just one simple qa task
@task
def longbench_v2() -> Task:

    # load dataset
    dataset = get_dataset(max_length=128000)
    
    return Task(
      dataset=dataset,
      solver=generate(),
      scorer=longbench_v2_scorer(),
      name="longbench_v2",
      config=GenerateConfig(temperature=0.1, max_tokens=128),
    )
    
