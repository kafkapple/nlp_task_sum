import pandas as pd
from pathlib import Path
from typing import List

def save_predictions(predictions: List[str], fnames: List[str], output_path: str):
    """Save predictions to CSV file"""
    output = pd.DataFrame({
        "fname": fnames,
        "summary": predictions
    })
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False) 