"""Utilities for fetching and saving OEIS sequences."""
import os
import json
import pickle
import requests
from pathlib import Path
from typing import List, Optional, Union, Dict

def fetch_oeis_sequence(a_number: str, max_terms: Optional[int] = None) -> List[int]:
    """Fetch an OEIS sequence by its A-number.
    
    Args:
        a_number: The A-number of the sequence (e.g. 'A001011')
        max_terms: Maximum number of terms to return. If None, returns all terms.
    
    Returns:
        List of integers representing the sequence
    """
    # Clean the A-number format
    a_number = a_number.upper().strip()
    if not a_number.startswith('A'):
        a_number = 'A' + a_number
    
    # Construct the URL
    url = f"https://oeis.org/search?q=id:{a_number}&fmt=json"
    
    # Make the request
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse the response
    data = response.json()
    if len(data) != 1:
        raise ValueError(f"Expected 1 result for {a_number}, got {len(data)}: response was {data}")

    single_sequence=data[0]
    
    if 'data' not in single_sequence or not single_sequence['data']:
        raise ValueError(f"No sequence found for {a_number}: response was {single_sequence}")
    # Get the data sequence
    sequence = single_sequence['data']
    
    # Convert to integers and limit if needed
    sequence = [int(x) for x in sequence.split(',')]
    if max_terms is not None:
        sequence = sequence[:max_terms]
    
    return sequence
def save_oeis_sequence(a_number: str, save_path: Optional[str] = None, max_terms: Optional[int] = None) -> str:
    """Fetch and save an OEIS sequence to pickle and json files.
    
    Args:
        a_number: The A-number of the sequence (e.g. 'A001011')
        save_path: Path to save the files. If None, uses ./examples/oeis_data/<seqname>.[pkl|json]
        max_terms: Maximum number of terms to save. If None, saves all terms.
    
    Returns:
        Tuple of (pickle_path, json_path, sequence)
    """
    # Get the sequence
    sequence = fetch_oeis_sequence(a_number, max_terms)
    
    # Clean the A-number format for filename
    a_number = a_number.upper().strip()
    if not a_number.startswith('A'):
        a_number = 'A' + a_number
    
    # Set up save path
    if save_path is None:
        save_path = os.path.join('examples', 'oeis_data')
    
    # Create directory if it doesn't exist
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Create full file paths
    pkl_path = os.path.join(save_path, f"{a_number}.pkl")
    json_path = os.path.join(save_path, f"{a_number}.json")
    
    # Save as pickle
    with open(pkl_path, 'wb') as f:
        pickle.dump(sequence, f)
        
    # Save as JSON
    with open(json_path, 'w') as f:
        json.dump(sequence, f)
    
    return pkl_path, json_path, sequence
