import numpy as np

def signed_sin(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the signed sine of the angle θ between vectors 'a' and 'b',
    with the sign determined by the direction of the given 'normal' vector.

    θ is the angle from 'a' to 'b' in the plane orthogonal to 'normal'.

    Returns:
        float: signed sine of the angle, in the range [-1, 1]
    """
    cross_product = np.cross(a, b)
    
    # Normalizzazione
    norm_product = np.linalg.norm(a) * np.linalg.norm(b)
    
    return float(cross_product / norm_product)