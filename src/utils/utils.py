import json
import numpy as np

def read_json_file(file_path):
    """
    Reads a JSON file and returns the data as a Python dictionary.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        dict: Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def quaternion_multiply(q1, Q2):
        # q1*Q2
        x0, y0, z0, w0 = q1
        
        return np.array([w0*Q2[:,0] + x0*Q2[:,3] + y0*Q2[:,2] - z0*Q2[:,1],
                         w0*Q2[:,1] + y0*Q2[:,3] + z0*Q2[:,0] - x0*Q2[:,2],
                         w0*Q2[:,2] + z0*Q2[:,3] + x0*Q2[:,1] - y0*Q2[:,0],
                         w0*Q2[:,3] - x0*Q2[:,0] - y0*Q2[:,1] - z0*Q2[:,2]]).T