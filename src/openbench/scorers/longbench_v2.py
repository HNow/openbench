import re
from inspect_ai.scorer import (
    accuracy,
    scorer,
    stderr,
    Score,
    Target,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState


#direct text comparison as done in paper
def extract_response(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None
    

@scorer(metrics=[accuracy(),stderr()])
def longbench_v2_scorer():
    """
    A scorer that evaluates answers in the format "The correct answer is (A)" or "The correct answer is A".
    """
    async def score(state: TaskState, target: Target) -> Score:
        response = extract_response(state.output.completion)
        #direct text comparison as done in paper
        is_correct = response == target.text.strip()
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=response,
            explanation=f"The scorer used is longbench_v2_scorer. The response is {response} and the target is {target.text}.",
        )
    return score