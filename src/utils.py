import pandas as pd
import lmstudio as lms
from lmstudio import PredictionResult
import textwrap
from pydantic import BaseModel
from pprint import pprint
from tqdm import tqdm
from pathlib import Path

def generate_mock_labels():
    """
    Use a LLM to generate mock labels for the clinical notes corpus for
    educational purposes only. These labels are not to be used for production systems.
    """

    # Instantiate a pretrained LLM through LM Studio
    model = lms.llm()

    # Prompt with specifications for the mock labels
    prompt = \
        textwrap.dedent("""
        You are a medical research assistant tasked with classifying the severity
        of the following clinical summary and doctor/patient dialogue with an
        integer score between 0 - 2, where 0 refers to the condition as stable
        or improving, 1 refers to new or worsening condition without hospitalisation,
        2 refers to any condition requiring hospitalisation or emergency treatment.
        For each score, concisely state the main reason in less than 20 words.
        """).strip()

    # We require the output to be structured
    class GeneratedSeverityScore(BaseModel):
        score:  int
        reason: str

    source_paths: list[str] = [
        "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_TrainingSet.csv",
        "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_ValidationSet.csv",
        "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_Final_200_TestSet_1.csv",
        "clinical_notes_corpus/data/mts-dialog/MTS_Dataset_Final_200_TestSet_2.csv"
    ]

    destination_paths: list[str] = [
        "mock_labels/mts-dialog/MTS_Dataset_TrainingSet.csv",
        "mock_labels/mts-dialog/MTS_Dataset_ValidationSet.csv",
        "mock_labels/mts-dialog/MTS_Dataset_Final_200_TestSet_1.csv",
        "mock_labels/mts-dialog/MTS_Dataset_Final_200_TestSet_2.csv"
    ]

    # Interate through each of the datasets
    for src_path, dst_path in tqdm(zip(source_paths, destination_paths), desc="Datasets"):

        # Check and create file paths where necessary
        assert Path(src_path).exists(), f"could not find source file: {src_path}"
        Path(dst_path).parent.mkdir(exist_ok=True, parents=True)

        # Read in the source data and create the labels.
        df = pd.read_csv(
            filepath_or_buffer = src_path,
            index_col          = 0
            )

        generated_scores = []
        for i, row in tqdm(df.iterrows(), desc="Samples"):
            summary  = row["section_text"]
            dialogue = row["dialogue"]

            result: PredictionResult = model.respond(
                history=f"{prompt} Summary: {summary} Dialogue: {dialogue}",
                config={ "temperature": 0.0 },
                response_format=GeneratedSeverityScore
            )
            assert isinstance(result.parsed, dict), "model output is not a dict"
            score = {
                "ID":     i,
                "score":  result.parsed["score"],
                "reason": result.parsed["reason"]
            }
            generated_scores.append(score)

        generated_scores_df = pd.DataFrame(generated_scores)
        generated_scores_df.to_csv(dst_path, index=False)

if __name__ == "__main__":
    generate_mock_labels()