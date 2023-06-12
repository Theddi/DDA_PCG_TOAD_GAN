# Code from https://github.com/Mawiszus/TOAD-GAN
import numpy as np

# Function to slice Map into multiple parts
class Mapslicer:

    def __init__(self, level_obj, iterations=12):
        self.name = level_obj.name
        self.level_mat = level_obj.ascii_level
        self.width = iterations * 2
        self.its = iterations

    '''
    slice_level slices the level input read by read_level into slices of size "width"
    Double the amount of slices will be created to better cover the whole map
    Therefore the level is iterated in "its" steps
    returns the path to the sliced files
    '''

    def slice_level(self):
        levelWidth = len(self.level_mat[0])
        levelHeight = len(self.level_mat)

        outputpath = OUT + self.name + "_SLICED/"
        pathExist = os.path.exists(outputpath)
        if not pathExist:
            os.makedirs(outputpath)

        # Create base level for time measurement
        baseLevel = []
        for h in range(levelHeight):
            if h < levelHeight-2:
                baseLevel.append("\n" + "-" * levelWidth if h != 0 else "" + "-" * (self.width-1))
            else:
                baseLevel.append("\n" + "X" * levelWidth if h != 0 else "" + "X" * (self.width-1))
        with open(outputpath + self.name + "_slice_base.txt", 'w') as file:
            file.writelines(baseLevel)

        # Create level slices
        for i in range(0, levelWidth - self.width, self.its):
            levelSlice = []
            for h in range(levelHeight):
                number = "%03d" % int(i / self.its)
                levelSlice.append("\n" + self.level_mat[h][i:i + self.width]
                                  if h != 0 else "" + self.level_mat[h][i:i + self.width])
            sliceFileName = self.name + "_slice" + number + ".txt"
            with open(outputpath + sliceFileName, 'w') as file:
                file.writelines(levelSlice)
        return os.path.abspath(outputpath)


# Miscellaneous functions to deal with ascii-token-based levels.
def load_level_from_text(path_to_level_txt):  # , replace_tokens=REPLACE_TOKENS):
    """ Loads an ascii level from a text file. """
    with open(path_to_level_txt, "r") as f:
        ascii_level = []
        for line in f:
            # for token, replacement in replace_tokens.items():
            #     line = line.replace(token, replacement)
            ascii_level.append(line)
    return ascii_level


def ascii_to_one_hot_level(level, tokens):
    """ Converts an ascii level to a full token level tensor. """
    oh_level = np.zeros((len(tokens), len(level), len(level[-1])))
    for i in range(len(level)):
        for j in range(len(level[-1])):
            token = level[i][j]
            if token in tokens and token != "\n":
                oh_level[tokens.index(token), i, j] = 1
    return oh_level


def one_hot_to_ascii_level(level, tokens):
    """ Converts a full token level tensor to an ascii level. """
    ascii_level = []
    for i in range(level.shape[2]):
        line = ""
        for j in range(level.shape[3]):
            line += tokens[level[:, :, i, j].argmax()]
        if i < level.shape[2] - 1:
            line += "\n"
        ascii_level.append(line)
    return ascii_level


def read_level_from_file(input_dir, input_name, tokens=None):  # , replace_tokens=REPLACE_TOKENS):
    """ Returns a full token level tensor from a .txt file. Also returns the unique tokens found in this level.
    Token. """
    txt_level = load_level_from_text("%s/%s" % (input_dir, input_name))  # , replace_tokens)
    uniques = set()
    for line in txt_level:
        for token in line:
            # if token != "\n" and token != "M" and token != "F":
            if token != "\n":  # and token not in replace_tokens.items():
                uniques.add(token)
    uniques = list(uniques)
    uniques.sort()  # necessary! otherwise we won't know the token order later
    oh_level = ascii_to_one_hot_level(txt_level, uniques if tokens is None else tokens)
    # return oh_level.unsqueeze(dim=0), uniques
    return np.expand_dims(oh_level, 0), uniques


solid_tokens = ['X', '#', 'S', '%', 't', '?', '@', '!', 'C', 'D', 'U', 'L']


def place_a_mario_token(level):
    """ Finds the first plausible spot to place Mario on. Especially important for levels with floating platforms.
    level is expected to be ascii."""
    # First check if default spot is available
    for j in range(1, 4):
        if level[-3][j] == '-' and level[-2][j] in solid_tokens:
            tmp_slice = list(level[-3])
            tmp_slice[j] = 'M'
            level[-3] = "".join(tmp_slice)
            return level

    # If not, check for first possible location from left
    for j in range(len(level[-1])):
        for i in range(1, len(level)):
            if level[i - 1][j] == '-' and level[i][j] in solid_tokens:
                tmp_slice = list(level[i - 1])
                tmp_slice[j] = 'M'
                level[i - 1] = "".join(tmp_slice)
                return level

    return level  # Will only be reached if there is no place to put Mario

def get_num_enemies(level):
    enemies = 0
    for line in level.ascii_level:
        enemies += line.count('g')
        enemies += line.count('k')
    return enemies


def is_token_stuck(level, height, length, begin):
    if level[height+1][length] in solid_tokens \
         and level[height-1][length] in solid_tokens \
         and level[height][length+1] in solid_tokens \
         and (level[height][length-1] in solid_tokens or length == begin):
        return True
    return False


def create_base_slice(height, length):
    # Create base level for time measurement
    baseLevel = []
    for h in range(height):
        if h < height - 2:
            baseLevel.append("-" * (length - 1) + "\n")
        else:
            baseLevel.append("X" * (length - 1) + "\n")
    return baseLevel


def place_token_with_limits(level, begin=None, end=None, token='M'):
    if token == 'M':
        # Check for first possible location from lower left
        for j in range(len(level[-1])):
            if (j < begin if begin else False) or (j > end if end else False):
                continue
            for i in reversed(range(1, len(level))):
                if level[i - 1][j] == '-' and level[i][j] in solid_tokens and not is_token_stuck(level, i-1, j, begin):
                    tmp_slice = list(level[i - 1])
                    tmp_slice[j] = 'M'
                    level[i - 1] = "".join(tmp_slice)
                    return level
    elif token == 'F':
        # Check for last possible location from lower right
        for j in reversed(range(len(level[-1]))):
            if (j < begin if begin else False) or (j > end if end else False):
                continue
            for i in reversed(range(1, len(level))):
                if level[i - 1][j] == '-':
                    tmp_slice = list(level[i - 1])
                    tmp_slice[j] = 'F'
                    level[i - 1] = "".join(tmp_slice)
                    return level
    return level
