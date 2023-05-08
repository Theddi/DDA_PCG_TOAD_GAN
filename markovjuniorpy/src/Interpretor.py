import re
from MarkovJunior import *
import xml.etree.ElementTree as ET


def parse_xml(path: str):
    """
    Parses the xml file in to a Context object
    """
    start_time = time.time()
    context = ET.parse(path)
    environment = ""
    grammar = []

    shuffleMatches = int(context.getroot().attrib["shuffle_matches"]) 
    seed = int(context.getroot().attrib["seed"])
    environment = parse_environment(context.find('environment').get("path"))
    
    # Get grammar
    for item in list(context.find('grammar')):
        if item.tag == 'sequence':
            grammar.append(parse_sequence(item))
        elif item.tag == 'markov':
            grammar.append(parse_markov(item))
        elif item.tag == 'one':
            grammar.append(parse_one(item))
        elif item.tag == 'all':
            grammar.append(parse_all(item))
        elif item.tag == 'prl':
            grammar.append(parse_prl(item))
            
    con = Context(environment, grammar, seed, shuffleMatches)
    end_time = time.time()
    print("Parsing grammar took:    ", (end_time - start_time), "seconds")
    return con


def make_legend(grammar):
    legend = ""
    return ""


def parse_one(one):
    res = OneMulti(rulesRandom=one.get("random"))
    for rule in one.findall('rule'):
        pattern = format_param(rule.get('in'))
        replacement = format_param(rule.get('out'))
        rotation = rule.get('rot')
        if len(pattern) != len(replacement) or len(pattern[0]) != len(replacement[0]):
            print("Dimension problem for rule: One")
            exit()
        var = Rule(pattern, replacement, rotation)
        res.addRule(var)
    return res


def parse_all(all):
    res = AllMulti(rulesRandom=all.get("random"))
    for rule in all.findall('rule'):
        pattern = format_param(rule.get('in'))
        replacement = format_param(rule.get('out'))
        rotation = rule.get('rot')
        if len(pattern) != len(replacement) or len(pattern[0]) != len(replacement[0]):
            print("Dimension problem for rule: All")
            exit()
        var = Rule(pattern, replacement,rotation)
        res.addRule(var)
    return res


def parse_prl(prl):
    res = PrlMulti(rulesRandom=prl.get("random"))
    for rule in prl.findall('rule'):
        pattern = format_param(rule.get('in'))
        replacement = format_param(rule.get('out'))
        rotation = rule.get('rot')
        if len(pattern) != len(replacement) or len(pattern[0]) != len(replacement[0]):
            print("Dimension problem for rule: Prl")
            exit()
        var = Rule(pattern, replacement, rotation)
        res.addRule(var)
    return res


def parse_markov(markov):
    res = Markov()
    for item in list(markov):
        if item.tag == 'sequence':
            res.addItem(parse_sequence(item))
        elif item.tag == 'one':
            res.addItem(parse_one(item))
        elif item.tag == 'all':
            res.addItem(parse_all(item))
        elif item.tag == 'prl':
            res.addItem(parse_prl(item))
    return res


def parse_sequence(sequence):
    res = Sequence(loop=sequence.get("loop"))
    for item in list(sequence):
        if item.tag == 'markov':
            res.addItem(parse_markov(item))
        elif item.tag == 'one':
            res.addItem(parse_one(item))
        elif item.tag == 'all':
            res.addItem(parse_all(item))
        elif item.tag == 'prl':
            res.addItem(parse_prl(item))
    return res


def parse_environment(path):
    with open(path, 'r') as file:
        file_contents = file.read()
    return file_contents


def format_param(param: str) -> str:
    return param.replace('/', '\n')

