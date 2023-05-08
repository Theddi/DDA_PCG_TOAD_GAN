import PatternMatch as pm


def getImage(name:str='output'):
    """
    Parses the output file as string
    """
    image = ""
    with open(name+'.txt', 'r') as file:
        image = file.read()
    
    return image

def extractRelations():
    """
    Finds relations between patterns
    """
    pass
 
def createRandomGrammar():
    """
    Creates random rules/rule sets using the found relations
    """
    pass

