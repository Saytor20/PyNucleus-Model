import pandas as pd
from ..rag.engine import ask
from ..utils.logger import logger

CSV_PATH = "data/validation/golden_dataset.csv"

def run_eval(threshold=0.7):
    df = pd.read_csv(CSV_PATH)
    correct = 0
    for idx, row in df.iterrows():
        answer = ask(row["question"])["answer"].lower()
        keywords = [kw.strip().lower() for kw in row["expected_keywords"].split(",")]
        if all(kw in answer for kw in keywords):
            correct += 1
        else:
            logger.warning(f"Q: {row['question']} missed keywords: {keywords}")

    score = correct / len(df)
    logger.info(f"Golden dataset accuracy: {score:.2%}")
    return score >= threshold 