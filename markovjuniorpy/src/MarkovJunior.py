import PatternMatch as pm
import numpy as np
import random
import time


class Context():
    """
    A context has an environment where a defined grammar can be applied
    """
    seed = 0
    seedCounter = 0
    allowMatchShuffle = False

    def __init__(self, environment: str = "", grammar=None, s=0, ams=False):
        global seed
        global allowMatchShuffle
        seed = s
        allowMatchShuffle = ams

        # Environment is a 2d String
        self.environment = environment

        # Grammar includes sequences or markov rule sets. It has left to right hiearchy.
        if grammar == None:
            self.grammar = []
        else:
            self.grammar = grammar

    def applyGrammar(self) -> str:
        """
        Apply the grammar on the environment, returns the result as string
        """
        global seedCounter
        seedCounter = 0
        result = None
        start_time = time.time()
        env = pm.strToNpArray(self.environment)

        for index, item in enumerate(self.grammar):
            print('Applying grammar element: ' ,index+1,"/",len(self.grammar))
            if isinstance(item, MultiRule) or isinstance(item, Rule):
                result = item.applyRule(env)
            elif isinstance(item, Markov) or isinstance(item, Sequence):
                result = item.applyRuleSet(env)
            print ("\033[A                             \033[A")
        end_time = time.time()

        print("Generation took:         ", (end_time - start_time), "seconds")
        return pm.npArrayToString(result[0])

    def addToGrammar(self, item):
        self.grammar.append(item)


class Rule():
    """
    Fundamental rule class, a rule consists of an input(pattern) and an output(replacement).
    A rule is applied on an environment (background) on the possible areas in the environment
    """

    def __init__(self, pattern: str, replacement: str, rotation=0):
        if rotation == None:
            self.rotation = 0
        else:
            self.rotation = int(rotation)
        
        self.pattern = np.rot90(pm.strToNpArray(pattern), self.rotation)
        self.replacement = np.rot90(pm.strToNpArray(replacement), self.rotation)


class MultiRule():
    """
    Secondary rule class, with multiple input(pattern) and output(replacement) tuples (rules).
    A rule is applied on an environment (background) on the possible areas in the environment.
    MultiRole has the advantage of delvering a rule from its rule list to be applied
    """
    
    def __init__(self, rules=None, rulesRandom=0):
        """
        If rules random is true only one random rule from the rules
        will be applied, else all the rules will be applied in iteration,
        default is iteration
        """
        if rules == None:
            self.rules = []
        else:
            self.rules = rules

        if rulesRandom == None:
            self.rulesRandom = False
        else:
            self.rulesRandom = int(rulesRandom)

    def addRule(self, rule: Rule):
        self.rules.append(rule)

    def applyRule(self, environment: str):
        print("selecting a random rule and returning the result")


class OneMulti(MultiRule):
    """
    First pattern occuerence is replaced
    """

    def applyRule(self, environment: str):
        """
        Apply rule to the first occuerence
        """
        global seedCounter
        background = None
        success = False
        if self.rulesRandom:
            random.Random(seed+seedCounter).shuffle(self.rules)
        seedCounter += 1
        for rule in self.rules:
            background, success = pm.replacePattern(seed+seedCounter,
                                                    rule, background=environment, count=1, allowMatchShuffle=allowMatchShuffle)
            if success and self.rulesRandom:
                return (background, success)
        return (background, success)


class AllMulti(MultiRule):
    """
    All pattern occuerences are replaced without overlapping
    """

    def applyRule(self, environment: str):
        """
        Apply rule to all occuerences without overlapping
        """
        global seedCounter
        background = None
        success = False
        if self.rulesRandom:
            random.Random(seed+seedCounter).shuffle(self.rules)
        seedCounter += 1
        for rule in self.rules:
            background, success = pm.replacePattern(seed+seedCounter,
                                                    rule, background=environment, overlap=False, allowMatchShuffle=allowMatchShuffle)
            if success and self.rulesRandom:
                return (background, success)
        return (background, success)


class PrlMulti(MultiRule):
    """
    All pattern occuerences are replaced by not caring about overlaps
    """

    def applyRule(self, environment: str):
        """
        Apply rule to all occuerences with overlapping
        """
        global seedCounter
        background = None
        success = False
        if self.rulesRandom:
            random.Random(seed+seedCounter).shuffle(self.rules)
        seedCounter += 1
        for rule in self.rules:
            background, success = pm.replacePattern(seed+seedCounter,
                                                    rule, background=environment, overlap=True, allowMatchShuffle=allowMatchShuffle)
            if success and self.rulesRandom:
                return (background, success)
        return (background, success)


class RuleSet():
    # This may include Markov or Rule type items
    def __init__(self, items=None):
        if items == None:
            self.items = []
        else:
            self.items = items


class Sequence(RuleSet):
    """
    Sequence applies rules or markov rule sets one time with hiearichal order
    left to right
    """

    def __init__(self, items=None, loop=1):
        if items == None:
            self.items = []
        else:
            self.items = items

        if loop == None:
            self.loop = 1
        else:
            self.loop = int(loop)

    def applyRuleSet(self, environment: str):
        success = False
        for i in range(self.loop):
            for item in self.items:
                if isinstance(item, MultiRule) or isinstance(item, Rule):
                    environment, success = item.applyRule(environment)
                elif isinstance(item, Markov):
                    environment, success = item.applyRuleSet(environment)
        return environment, success

    def addItem(self, item):
        self.items.append(item)


class Markov(RuleSet):
    """
    Markov algorithm applies rules or sequences
    """

    def applyRuleSet(self, environment: str):
        success = True
        while success:
            for item in self.items:
                if isinstance(item, MultiRule) or isinstance(item, Rule):
                    environment, success = item.applyRule(environment)
                elif isinstance(item, Sequence):
                    environment, success = item.applyRuleSet(environment)
                if success:               
                    # print(pm.npArrayToString(environment)+'\n')
                    break

        return environment, success

    def addItem(self, item):
        self.items.append(item)

    