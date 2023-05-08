import math
import numpy as np
import string
import random
import cv2
from numba import jit


def matchPattern(pattern, background, wildcard='*'):
    """
    Finds matching patterns on a background, returns the upper left coordinates in form
    (y,x) of each found pattern

    :param np.array pattern:        to be searched pattern
    :param np.array background:     searching environment
    :param char wildcard:           the wildcard ascii character
    """

    # Create a mask array to ignore wildcard values
    mask = np.where(pattern != ord(wildcard), True, False).astype(np.uint8)

    # Apply matchTemplate to the encoded arrays
    result = cv2.matchTemplate(background, pattern, cv2.TM_SQDIFF_NORMED, mask=mask)

    # Get only the near perfect matches
    matches = np.where(result <= 0.0001)

    # Convert perfect machtes to coordinates
    matches = zip(matches[1], matches[0])
    
    # Get only the ones inbounds of the background
    inBounds = []
    for x, y in matches:
        if not isRectangleOutOfBounds(x, y, len(pattern[0]), len(pattern), background):
            inBounds.append((x, y))
    matches = inBounds
    return matches


def replacePattern(seed, rule, background, overlap=False, count=math.inf, wildcard='*', allowMatchShuffle=False):
    """
    Replaces matched patterns and returns the new background

    :param np.array pattern:        to be searched pattern
    :param np.array replacement:    to be replaced pattern
    :param np.array background:     searching environment
    :param bool overlap:            allow overlaps when replacing
    :param int count:               how many occurrences must be replaced
    :param char wildcard:           the wildcard ascii character

    :returns updated background as np.array and true if a replacement happened
    """

    pattern = rule.pattern
    replacement = rule.replacement

    matches = matchPattern(pattern, background)

    if len(matches) > 0:
        if allowMatchShuffle:
            random.Random(seed).shuffle(matches)
    else:
        return (background, False)

    if count > len(matches):
        count = len(matches)

    replaced = False
    for i in range(count):
        x, y = matches[i]
        window = background[y:y+len(pattern), x:x+len(pattern[0])]

        # If pattern overlap is not allowed, current window must be a valid match
        if overlap or comparePatterns(pattern, window):
            # Replace the matched pattern except the places with the wild cards
            mask = np.where(replacement != ord(wildcard), True, False)
            background[y:y+len(pattern), x:x+len(pattern[0])
                       ][mask] = replacement[mask]
            replaced = True

    return (background, replaced)


def strToNpArray(pattern):
    """
    Converts 2d string to 2d np.array, helper method
    """
    if isinstance(pattern, str):
        result = []
        for row in pattern.strip().split('\n'):
            result.append(list(ord(element) for element in row.strip()))
        return np.array(result, dtype='uint8')
    return pattern


def npArrayToString(pattern):
    """
    Converts 2d np.array to 2d string, helper method 
    """
    result = ""
    for index, row in enumerate(pattern):
        result += ("".join(list(chr(element) for element in row)))
        if index < len(pattern)-1:
            result += "\n"
    return result


@jit(nopython=True)
def comparePatterns(template, window, wildcard='*'):
    """
    Checks if two patterns are equivalent 
    """
    # Replace wildcard values if any in template with corresponding values in background
    template = np.where(template == ord(wildcard), window, template)
    return np.array_equal(template, window)


@jit(nopython=True)
def isRectangleOutOfBounds(x, y, width, height, background):
    # Check if rectangle is outside of background
    if x < 0 or y < 0 or x + width > len(background[0]) or y + height > len(background):
        return True
    return False