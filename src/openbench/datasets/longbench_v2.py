from inspect_ai.dataset import Sample, Dataset, hf_dataset
from openbench.utils.text import get_token_count,get_gemma_tok_cnt



zero_shot_template = """Please read the following text and answer the question below.

<text>
{DOC}
</text>

What is the correct answer to this question: {question}
Choices:
(A) {C_A}
(B) {C_B}
(C) {C_C}
(D) {C_D}

Format your response as follows: "The correct answer is (insert answer here)"."""

def record_to_sample(record: dict) -> Sample:
    # step 1 build full input following 0shot template above
    prompt=zero_shot_template.format(
        DOC=record["context"].strip(),
        question=record["question"],
        C_A=record["choice_A"],
        C_B=record["choice_B"],
        C_C=record["choice_C"],
        C_D=record["choice_D"],
    )
    
    return Sample(
        input=prompt,
        target=record["answer"],
        metadata={
            "question": record["question"],
            "choice_A": record["choice_A"],
            "choice_B": record["choice_B"],
            "choice_C": record["choice_C"],
            "choice_D": record["choice_D"],
            "domain": record["domain"],
            "sub_domain": record["sub_domain"],
            "length": record["length"],
        },
    )

def filter_dataset_by_task(dataset: Dataset, task: str) -> Dataset:
    return dataset.filter(lambda x: x.metadata["domain"] == task)


def filter_dataset_by_token_count(dataset: Dataset, max_length: int) -> Dataset:
    # filter out samples that are longer than max_length
    return dataset.filter(lambda x: get_gemma_tok_cnt(x.input) <= max_length)

    

def get_dataset(max_length: int = None):
    """
    Load in the LongBench v2 dataset and trim it to only 1 task for now
    """
    # if we only had to remap variable names we use FieldSpec, but its more involved here with building the full input
    # so instead we provide the full function to build inputs!
    dataset = hf_dataset("zai-org/LongBench-v2", split="train",sample_fields=record_to_sample)
    # now we need to apply filtering, since the dataset is an object this should work as a property/method
    
    dataset = filter_dataset_by_task(dataset, "Single-Document QA")
    if max_length is not None:
        dataset = filter_dataset_by_token_count(dataset, max_length)
    return dataset
