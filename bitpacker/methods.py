"""
Functions for bit manipulation
"""
import numpy as np

__all__ = []

__all__ += ['get_bits']

def get_bits(number, a, b):
    """
    Extracts bits from a specific range (a through b) from the given number.
    Assumes that the least significant bit is at position 0.

    :param number: The number from which to extract the bits.
    :param a: The starting position of the bit range (inclusive).
    :param b: The ending position of the bit range (inclusive).
    :return: The extracted bits.

    Example:
    number = 0b110101100  # Example binary number
    a, b = 2, 5  # Range from bit 2 to bit 5
    result = get_bits(number, a, b)

    """
    # Create a mask with bits set in the range a through b
    mask = (1 << (b - a + 1)) - 1
    # Shift the number right by 'a' and apply the mask
    return (number >> a) & mask

__all__ += ['set_bits']
def set_bits(originals, bits_to_insert, positions, lengths):
    """
    Vectorized function to insert bits into a NumPy array of integers at specified positions.

    :param originals: A NumPy array of original integers.
    :param bits_to_insert: A NumPy array of integers representing bits to be inserted.
    :param positions: A NumPy array of positions at which to insert bits.
    :param lengths: A NumPy array of lengths indicating the number of bits to be inserted.
    :return: A NumPy array with the bits inserted.
    """
    # Create masks to clear the target areas in the original numbers
    masks = ((1 << lengths) - 1) << positions
    cleared_originals = originals & ~masks

    # Align the bits to be inserted
    aligned_bits = (bits_to_insert & ((1 << lengths) - 1)) << positions

    # Combine the cleared numbers with the aligned bits
    return cleared_originals | aligned_bits

__all__ += ['check_bits']
def check_bits(numbers, a, length, C):
    """
    Check if the bits from position a for a length in each number of a NumPy array are equal to a given value C.

    :param numbers: NumPy array of integers.
    :param a: Starting bit position.
    :param length: Length of the bit range.
    :param C: The value to compare the bit range against.
    :return: A boolean NumPy array where each element indicates whether the specified bit range in the corresponding number equals C.

    # Example usage
    numbers = np.array([0b1010, 0b1100, 0b1110, 0b0111])  # Example array of integers
    a = 1        # Starting bit position
    length = 2   # Length of the bit range
    C = 0b11     # Value to compare against (binary 11, which is 3 in decimal)
    """
    # Create a mask for the bit range
    mask = ((1 << length) - 1) << a

    # Shift C to align with the bit range
    C_aligned = C << a

    # Apply the mask and compare to the aligned C value
    return (numbers & mask) == C_aligned


__all__ += ['show_agents']
def show_agents(agents, n=5):    
    print('---')    
    cnt = 0
    for a in agents:
        cnt += 1
        print(bin(a))
        if cnt == n:
            break
    print('---')
