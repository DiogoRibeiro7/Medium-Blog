def find_max(numbers):
    """
    Find the maximum value in a list of numbers.

    Args:
        numbers (list): List of numbers.

    Returns:
        int: The maximum value in the list.

    >>> find_max([1, 5, 2, 8, 3])
    8
    >>> find_max([-3, -7, -1])
    -1
    """
    return max(numbers)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
