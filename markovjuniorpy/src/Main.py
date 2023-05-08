import argparse
import Displayer as dp
import Interpretor as interpretor
import EnvMaker as envm
from MarkovJunior import *

envm.make_env(400,100,"B")

def updateWindow(context):
    env = context.applyGrammar()
    dp.root.update_idletasks()
    dp.root.update()
    dp.display_image(env)

def updateOutput(context):
    env = context.applyGrammar()
    #print(env)
    dp.save_as_png(env)
    dp.save_as_txt(env)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute a markovjunior grammar')
    parser.add_argument('filename', help='the name of the markovjunior grammar')
    args = parser.parse_args()
    context = interpretor.parse_xml(args.filename)
    #while True:
    updateOutput(context=context)

