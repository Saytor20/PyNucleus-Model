import pandas as pd
from ..rag.engine import ask
from ..utils.logger import logger

CSV_PATH = "data/validation/golden_dataset.csv"

def run_eval(threshold=0.7, sample_size=None):
    df = pd.read_csv(CSV_PATH)
    
    # Random sampling if sample_size is specified
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)  # Fixed seed for reproducibility
        logger.info(f"Randomly sampled {sample_size} questions from {len(pd.read_csv(CSV_PATH))} total questions")
    
    correct = 0
    for idx, row in df.iterrows():
        answer = ask(row["question"])["answer"].lower()
        keywords = [kw.strip().lower() for kw in row["expected_keywords"].split(",")]
        if all(kw in answer for kw in keywords):
            correct += 1
        else:
            logger.warning(f"Q: {row['question']} missed keywords: {keywords}")

    score = correct / len(df)
    logger.info(f"Golden dataset accuracy: {score:.2%} ({correct}/{len(df)} questions)")
    return score >= threshold 