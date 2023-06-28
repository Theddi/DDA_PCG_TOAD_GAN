import tkinter.ttk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import ImageTk, Image, ImageDraw
from py4j.java_gateway import JavaGateway
import os
import platform
import time
import threading
import queue
import torch
import math
import sys

from utils.scrollable_image import ScrollableImage
from utils.tooltip import Tooltip
from utils.level_utils import read_level_from_file, one_hot_to_ascii_level, place_a_mario_token, \
    place_token_with_limits, create_base_slice
from utils.level_image_gen import LevelImageGen
from utils.toad_gan_utils import load_trained_pyramid, generate_sample, TOADGAN_obj
from wfc_2019f.wfc import wfc_control

from filelock import FileLock
import shutil
import json
import pandas as pd
import numpy as np

# PATHS
CURDIR = os.path.abspath(os.path.curdir)
# Path to the AI Framework jar for Playing levels
MARIO_AI_PATH = os.path.join(CURDIR, "Mario-AI-Framework/mario-1.0-SNAPSHOT.jar")
MARIO_AI_PATH_NEW = os.path.join(CURDIR, "Mario-AI-Framework/Mario-AI-FrameworkImproved.jar")
# Output folder
OUT = os.path.join(CURDIR, "tmp/")
pathExist = os.path.exists(OUT)
if not pathExist:
    os.makedirs(OUT)
# Path for game results
GAME_RESULT_PATH = os.path.join(OUT, "results.txt")
# Lock file extension
LOCK_EXT = ".lock"

# Check if windows user
if platform.system() == "Windows":
    import ctypes

    # Set Taskbar Icon
    my_appid = u'toad-gui.1.0'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(my_appid)

    # Make DPI aware - if size is too small or weird turn on/off
    # ctypes.windll.shcore.SetProcessDpiAwareness(True)


# Validate Function for Entries
def on_validate(in_str, act_type):
    if act_type == '1':  # insertion
        if not in_str.isdigit():
            return False
    return True


genCounter = 0
genName = ""


# Object holding important data about the current level
class LevelObject:
    def __init__(self, ascii_level, oh_level, image, tokens, scales, noises, name=""):
        self.ascii_level = ascii_level
        self.oh_level = oh_level  # one-hot encoded
        self.image = image
        self.tokens = tokens
        self.scales = scales
        self.noises = noises
        self.name = name

    def save(self, into):
        into.oh_level = self.oh_level
        into.ascii_level = self.ascii_level
        into.name = self.name
        into.image = self.image
        into.scales = self.scales
        into.noises = self.noises
        into.tokens = self.tokens

    def restore(self, from_):
        self.oh_level = from_.oh_level
        self.ascii_level = from_.ascii_level
        self.name = from_.name
        self.image = from_.image
        self.scales = from_.scales
        self.noises = from_.noises
        self.tokens = from_.tokens

# Main GUI code
def TOAD_GUI():
    # Init Window
    root = Tk(className=" TOAD-GUI")

    # Thread Functions to keep GUI alive when playing/loading/generating
    class ThreadedClient(threading.Thread):
        def __init__(self, que, fcn, *args):
            threading.Thread.__init__(self)
            self.que = que
            self.fcn = fcn
            self.args = args

        def run(self):
            time.sleep(0.01)
            if len(self.args) > 0:
                self.que.put(self.fcn(*self.args))
            else:
                self.que.put(self.fcn())

    def spawn_thread(que, fcn, *args):
        thread = ThreadedClient(que, fcn, *args)
        thread.start()
        periodic_call(thread)
        return thread

    def periodic_call(thread):
        if thread.is_alive():
            root.after(100, lambda: periodic_call(thread))

    # Load Icons
    toad_icon = ImageTk.PhotoImage(Image.open('icons/toad_icon.png'))
    banner_icon = ImageTk.PhotoImage(Image.open('icons/banner.png'))
    load_level_icon = ImageTk.PhotoImage(Image.open('icons/folder_level.png'))
    load_generator_icon = ImageTk.PhotoImage(Image.open('icons/folder_toad.png'))
    generate_level_icon = ImageTk.PhotoImage(Image.open('icons/gear_toad.png'))
    play_level_icon = ImageTk.PhotoImage(Image.open('icons/play_button.png'))
    play_ai_level_icon = ImageTk.PhotoImage(Image.open('icons/play_ai_button.png'))
    iterate_level_icon = ImageTk.PhotoImage(Image.open('icons/iterate_button.png'))
    extract_slices_icon = ImageTk.PhotoImage(Image.open('icons/extract_slices.png'))
    save_level_icon = ImageTk.PhotoImage(Image.open('icons/save_button.png'))

    root.iconphoto(False, toad_icon)

    # Level to Image renderer
    ImgGen = LevelImageGen(os.path.join(os.path.join(os.curdir, "utils"), "sprites"))

    # Get images of tokens for edit context menu
    full_token_list = torch.load("files/token_list.pth")
    token_img_dict = {}
    for token in full_token_list:
        token_img_dict[token] = ImageTk.PhotoImage(ImgGen.render([token]))

    # Placeholder image for the preview
    placeholder = Image.new('RGB', (890, 256), (255, 255, 255))
    draw = ImageDraw.Draw(placeholder)
    draw.text((356, 128), "Level Preview will appear here.", (0, 0, 0))
    levelimage = ImageTk.PhotoImage(placeholder)

    # Define Variables
    level_obj = LevelObject('-', None, levelimage, ['-'], None, None)
    level_obj_tmp = LevelObject('-', None, levelimage, ['-'], None, None)
    toadgan_obj = TOADGAN_obj(None, None, None, None, None, None)

    level_l = IntVar()
    level_h = IntVar()
    load_string_gen = StringVar()
    load_string_txt = StringVar()
    level_path = StringVar()
    error_msg = StringVar()
    use_gen = BooleanVar()
    is_loaded = BooleanVar()
    q = queue.Queue()

    current_completion = IntVar()

    # Set values
    current_completion.set(1)
    level_l.set(0)
    level_h.set(0)
    load_string_gen.set("Click the buttons to open a level or generator.")
    load_string_txt.set(os.path.join(os.curdir, "levels"))
    level_path.set(False)
    error_msg.set("No Errors")
    use_gen.set(False)
    is_loaded.set(False)

    # ---------------------------------------- Define Callbacks ----------------------------------------
    def load_level():
        fname = fd.askopenfilename(title='Load Level', initialdir=os.path.join(os.curdir, 'levels'),
                                   filetypes=[("level .txt files", "*.txt")])
        load_level_by_path(fname)

    def load_level_by_path(fname):
        if len(fname) == 0:
            return  # loading was cancelled
        try:
            error_msg.set("Loading level...")
            is_loaded.set(False)
            use_gen.set(False)

            if fname[-3:] == "txt":
                load_string_gen.set('Path: ' + fname)
                level_path.set(fname)
                folder, lname = os.path.split(fname)
                level_obj.name = lname.replace(".txt", "")

                # Load level
                lev, tok = read_level_from_file(folder, lname)

                level_obj.oh_level = torch.Tensor(lev)  # casting to Tensor to keep consistency with generated levels
                level_obj.ascii_level = one_hot_to_ascii_level(lev, tok)

                # Check if a Mario token exists - if not, we need to place one
                m_exists = False
                for line in level_obj.ascii_level:
                    if 'M' in line:
                        m_exists = True
                if not m_exists:
                    level_obj.ascii_level = place_a_mario_token(level_obj.ascii_level)
                level_obj.tokens = tok

                img = ImageTk.PhotoImage(ImgGen.render(level_obj.ascii_level))
                level_obj.image = img

                level_obj.scales = None
                level_obj.noises = None

                level_l.set(lev.shape[-1])
                level_h.set(lev.shape[-2])

                is_loaded.set(True)
                use_gen.set(False)
                error_msg.set("Level loaded")
            else:
                error_msg.set("No level file selected.")
        except Exception:
            error_msg.set("No level file selected.")
        return

    def load_generator():
        fname = fd.askdirectory(title='Load Generator Directory', initialdir=os.path.join(os.curdir, 'generators'))

        if len(fname) == 0:
            return  # loading was cancelled
        try:
            error_msg.set("Loading generator...")
            use_gen.set(False)
            is_loaded.set(False)
            load_string_gen.set('Path: ' + fname)
            folder = fname
            global genName
            genName = os.path.split(fname)[1]

            # Load TOAD-GAN
            loadgan, msg = load_trained_pyramid(folder)
            toadgan_obj.Gs = loadgan.Gs
            toadgan_obj.Zs = loadgan.Zs
            toadgan_obj.reals = loadgan.reals
            toadgan_obj.NoiseAmp = loadgan.NoiseAmp
            toadgan_obj.token_list = loadgan.token_list
            toadgan_obj.num_layers = loadgan.num_layers

            level_l.set(toadgan_obj.reals[-1].shape[-1])
            level_h.set(toadgan_obj.reals[-1].shape[-2])

            error_msg.set(msg)

            is_loaded.set(False)
            use_gen.set(True)

        except Exception:
            error_msg.set("Could not load generator. Confirm that all needed .pth files are in the chosen folder.")

        return

    def save_txt(current=False, name=level_obj.name):
        if not current:
            save_name = fd.asksaveasfile(title='Save level (.txt/.png)', initialdir=os.path.join(os.curdir, "levels"),
                                         mode='w', defaultextension=".txt",
                                         filetypes=[("txt files", ".txt"), ("png files", ".png")])
            if save_name is None:
                return  # saving was cancelled
            elif save_name.name[-3:] == "txt":
                text2save = ''.join(level_obj.ascii_level)
                save_name.write(text2save)
                save_name.close()
            elif save_name.name[-3:] == "png":
                ImgGen.render(level_obj.ascii_level).save(save_name.name)
            else:
                error_msg.set("Could not save level with this extension. Supported: .txt, .png")
            return
        else:
            genFilePath = os.path.join(OUT, name + ".txt")
            with open(genFilePath, 'w') as save_name:
                text2save = ''.join(level_obj.ascii_level)
                save_name.write(text2save)
            return genFilePath

    def generate():
        global genCounter
        genCounter += 1
        if toadgan_obj.Gs is None:
            error_msg.set("Generator did not load correctly. Are all necessary files in the folder?")
        else:
            error_msg.set("Generating level...")
            is_loaded.set(False)
            # Get scaling from height and length
            sc_h = level_h.get() / toadgan_obj.reals[-1].shape[-2]
            sc_l = level_l.get() / toadgan_obj.reals[-1].shape[-1]

            # Generate level
            level_obj.name = genName + "_C" + str(genCounter)

            level, scales, noises = generate_sample(toadgan_obj.Gs, toadgan_obj.Zs, toadgan_obj.reals,
                                                    toadgan_obj.NoiseAmp, toadgan_obj.num_layers,
                                                    toadgan_obj.token_list,
                                                    scale_h=sc_l, scale_v=sc_h)
            level_obj.oh_level = level.cpu()
            level_obj.scales = scales
            level_obj.noises = noises

            level_obj.ascii_level = one_hot_to_ascii_level(level, toadgan_obj.token_list)
            redraw_image()

            is_loaded.set(True)
            set_button_state(None, None, None)
            print("Level generated!")
            error_msg.set("Level generated!")
        return

    def redraw_image(edit_mode=False, rectangle=[(0, 0), (16, 16)], scale=0):
        if is_loaded.get():
            # Check if a Mario token exists - if not, we need to place one
            m_exists = False
            f_exists = False
            for line in level_obj.ascii_level:
                if 'M' in line:
                    m_exists = True
                if 'F' in line:
                    f_exists = True
            if not m_exists:
                level_obj.ascii_level = place_token_with_limits(level_obj.ascii_level)[0]
            if not f_exists:
                level_obj.ascii_level = place_token_with_limits(level_obj.ascii_level, token='F')[0]

            if use_gen.get():
                level_obj.tokens = toadgan_obj.token_list

            # Render Image
            pil_img = ImgGen.render(level_obj.ascii_level)

            if edit_mode:
                # Add numbers to rows and cols
                l_draw = ImageDraw.Draw(pil_img)
                for y in range(level_obj.oh_level.shape[-2]):
                    l_draw.multiline_text((1, 4 + y * 16), str(y), (255, 255, 255),
                                          stroke_width=-1, stroke_fill=(0, 0, 0))
                for x in range(level_obj.oh_level.shape[-1]):
                    l_draw.multiline_text((6 + x * 16, 0), "".join(["%s\n" % c for c in str(x)]), (255, 255, 255),
                                          stroke_width=-1, stroke_fill=(0, 0, 0), spacing=0,
                                          align='right')
                # Add bounding box rectangle
                l_draw.rectangle(rectangle, outline=(255, 0, 0), width=2)

                # Affected tokens rectangle
                n_pads = len(level_obj.scales) - scale
                if n_pads > 0:  # if scale is chosen too big, this just does not render
                    padding_effect = 3 * n_pads
                    sc = level_obj.scales[scale].shape[-1] / level_obj.oh_level.shape[-1]
                    scaling_effect = math.ceil((1 / sc - 1) / 2)  # affected tokens in every direction
                    aoe = (padding_effect + scaling_effect) * 16
                    affected_rect = [(rectangle[0][0] - aoe, rectangle[0][1] - aoe),
                                     (rectangle[1][0] + aoe, rectangle[1][1] + aoe)]
                    l_draw.rectangle(affected_rect, outline=(255, 255, 0), width=2)

            # Cast image
            img = ImageTk.PhotoImage(pil_img)

            level_obj.image = img
            image_label.change_image(level_obj.image)

    def play_level(ai_select="Human", loop=False, playtime=30, visuals=True,
                   multicall=False, baselevel=None, slice_name=level_obj.name):
        ai = ai_select
        if not multicall:
            error_msg.set("Playing level...")
            is_loaded.set(False)
            remember_use_gen = use_gen.get()
            use_gen.set(False)

        # Py4j Java bridge uses Mario AI Framework
        gateway = JavaGateway.launch_gateway(classpath=MARIO_AI_PATH_NEW, die_on_exit=True,
                                             redirect_stdout=sys.stdout,
                                             redirect_stderr=sys.stderr,
                                             javaopts=["-Xms128m", "-Xmx256m"])

        agent = gateway.jvm.agents.robinBaumgarten.Agent()
        if ai in advanced_ais:
            game = gateway.jvm.mff.agents.common.AgentMarioGame()
        else:
            game = gateway.jvm.engine.core.MarioGame()
        try:
            if ai == "":
                pass
            elif ai in base_ais:
                if ai == "andySloane":
                    agent = gateway.jvm.agents.andySloane.Agent()
                if ai == "doNothing":
                    agent = gateway.jvm.agents.doNothing.Agent()
                if ai == "glennHartmann":
                    agent = gateway.jvm.agents.glennHartmann.Agent()
                if ai == "michal":
                    agent = gateway.jvm.agents.michal.Agent()
                if ai == "random":
                    agent = gateway.jvm.agents.random.Agent()
                if ai == "sergeyKarakovskiy":
                    agent = gateway.jvm.agents.sergeyKarakovskiy.Agent()
                if ai == "sergeyPolikarpov":
                    agent = gateway.jvm.agents.sergeyPolikarpov.Agent()
                if ai == "spencerSchumann":
                    agent = gateway.jvm.agents.spencerSchumann.Agent()
                if ai == "trondEllingsen":
                    agent = gateway.jvm.agents.trondEllingsen.Agent()
                if ai == "robinBaumgarten":
                    agent = gateway.jvm.agents.robinBaumgarten.Agent()
            elif ai in advanced_ais:
                if ai == "astar":
                    agent = gateway.jvm.mff.agents.astar.Agent()
                if ai == "astarDistanceMetric":
                    agent = gateway.jvm.mff.agents.astarDistanceMetric.Agent()
                if ai == "astarFast":
                    agent = gateway.jvm.mff.agents.astarFast.Agent()
                if ai == "astarGrid":
                    agent = gateway.jvm.mff.agents.astarGrid.Agent()
                if ai == "astarJump":
                    agent = gateway.jvm.mff.agents.astarJump.Agent()
                if ai == "astarPlanning":
                    agent = gateway.jvm.mff.agents.astarPlanning.Agent()
                if ai == "astarPlanningDynamic":
                    agent = gateway.jvm.mff.agents.astarPlanningDynamic.Agent()
                if ai == "astarWaypoints":
                    agent = gateway.jvm.mff.agents.astarWaypoints.Agent()
                if ai == "astarWindow":
                    agent = gateway.jvm.mff.agents.astarWindow.Agent()
                if ai == "robinBaumgartenSlim":
                    agent = gateway.jvm.mff.agents.robinBaumgartenSlim.Agent()
                if ai == "robinBaumgartenSlimImproved":
                    agent = gateway.jvm.mff.agents.robinBaumgartenSlimImproved.Agent()
                if ai == "robinBaumgartenSlimWindowAdvance":
                    agent = gateway.jvm.mff.agents.robinBaumgartenSlimWindowAdvance.Agent()
            else:
                agent = gateway.jvm.agents.human.Agent()
            oneLoop = True
            while loop or oneLoop:
                if baselevel:
                    result = game.runGame(agent, ''.join(baselevel), playtime, 0, visuals, 30, 2.0)
                else:
                    result = game.runGame(agent, ''.join(level_obj.ascii_level), playtime, 0, visuals, 30, 2.0)

                perc = int(result.getCompletionPercentage() * 100)
                current_completion.set(perc)

                info_list = [slice_name, ai, result.getCompletionPercentage(),
                             playtime - (result.getRemainingTime() / 1000), playtime]

                with FileLock(GAME_RESULT_PATH + LOCK_EXT):
                    with open(GAME_RESULT_PATH, 'a') as file:
                        # file.write(format_result_tostring(True, *info_list))
                        json.dump(info_list, file)
                        file.write("\n")
                if not multicall:
                    error_msg.set("Level Played. Completion Percentage: %d%%" % perc)
                else:
                    error_msg.set("Iterating: " + ai + " completion percentage: %d%%" % perc)
                oneLoop = False

        except Exception:
            if not multicall:
                error_msg.set("Level Play was interrupted.")
                is_loaded.set(True)
                use_gen.set(remember_use_gen)
            else:
                error_msg.set("Iterating: Level Play was interrupted.")

        finally:
            pid = gateway.java_process.pid
            gateway.shutdown()
            # TODO Generalize for other os than windows
            os.system("taskkill /f /pid %d /t" % pid)

            if not multicall:
                is_loaded.set(True)
                use_gen.set(remember_use_gen)  # only set use_gen to True if it was previously
        return

    def get_difficulty(resultDataframe):
        min_completion = 1/slice_length_var.get()
        new_dataframe = pd.DataFrame()

        grouped = resultDataframe.groupby('file_name')
        for name, group in grouped:
            # Check if random ai can complete slice and remove result
            randPerc = group.loc[group['agent'] == 'random', 'completion_percentage'].iloc[0]
            group = group.drop(group[group['agent'] == 'random'].index)

            # Completion modifier, penalty on less comletion due to exponent
            x = [min_completion, 1.0]  # Minimum completion of 1 Token
            y = [1, 0.5]
            completion_multiplier = np.interp(randPerc, x, y)**2
            group['completion_factor'] = (1.0 + min_completion - group['completion_percentage']) * completion_multiplier
            group['time_factor'] = group['time_needed'] / group['total_time']

            # Difficulty completion dependent on completion rate and time
            group['difficulty_score'] = group['completion_factor'] + group['time_factor']

            new_dataframe = pd.concat([new_dataframe, group])

        new_dataframe.to_csv(GAME_RESULT_PATH)

        # Calculate mean
        new_dataframe = new_dataframe[['file_name', 'difficulty_score']].groupby('file_name').mean()

        return new_dataframe

    def print_results(results):
        for res in results:
            print(format_result_tostring(False, *res))

    def format_result_tostring(newline=True, *args):
        format_str = ", ".join([f"{args[0]:<25}", f"{args[1]:<25}"])
        for i in args[2:]:
            format_str += ", " + '{0: <10}'.format(format(i, "10.3f")) if isinstance(i, float) \
                else ", " + '{0: <3}'.format(i)
        if newline:
            format_str += '\n'
        return format_str

    def delete_mario_finish():
        mar_fin = [None, None]
        for height, line in enumerate(level_obj.ascii_level):
            for length, tok in enumerate(line):
                if tok == 'M' or tok == 'F':
                    tmp_slice = list(level_obj.ascii_level[height])
                    tmp_slice[length] = '-'
                    level_obj.ascii_level[height] = "".join(tmp_slice)
                if tok == 'M':
                    mar_fin[0] = (height, length)
                if tok == 'F':
                    mar_fin[1] = (height, length)
        return mar_fin

    def set_mario_finish(mar_fin):
        for height, line in enumerate(level_obj.ascii_level):
            for length, tok in enumerate(line):
                if height == mar_fin[0][0] and length == mar_fin[0][1]:
                    tmp_slice = list(level_obj.ascii_level[height])
                    tmp_slice[length] = 'M'
                    level_obj.ascii_level[height] = "".join(tmp_slice)
                if height == mar_fin[1][0] and length == mar_fin[1][1]:
                    tmp_slice = list(level_obj.ascii_level[height])
                    tmp_slice[length] = 'F'
                    level_obj.ascii_level[height] = "".join(tmp_slice)
        return mar_fin

    def ai_iterate_level(clear="all"):
        # Set variables
        remGen = use_gen.get()
        is_loaded.set(False)
        editmode.set(False)
        threads = []
        standard_agent_time = 10
        error_msg.set("Iterating...")
        da_progressbar['value'] = 0

        # Testing full level with strongest ai, to check if it's completable
        current_difficulty_label.config(text="Current Difficulty: playtesting...")
        testthread = spawn_thread(q, play_level, "astar", False, 60, True, True)
        testthread.join()

        current_difficulty_label.config(text="Current Difficulty: determining...")
        if clear == "all" and os.path.isfile(GAME_RESULT_PATH):
            shutil.rmtree(OUT, ignore_errors=True)
        if clear == "results" and os.path.isfile(GAME_RESULT_PATH):
            f = open(GAME_RESULT_PATH, 'r+')
            f.truncate(0)  # need '0' when using r+

        if current_completion.get() < 100:
            current_difficulty_label.config(text="Current Difficulty: Error")
            error_msg.set("Level is not completable!")
            is_loaded.set(True)
            return

        # Determines slices by setting Mario and Finish position in level_obj, and iterates those with AI before reset
        slices = []  # List of tuples with Mario and Flag position [(Mario, Flag),..] each (row, column) in ascii level
        levelHeight = len(level_obj.ascii_level)
        levelWidth = len(level_obj.ascii_level[0])
        sl = slice_length_var.get()

        # Take original Mario and Finish position out of level
        marfin = delete_mario_finish()

        # Determine and play level slices
        sliceCounter = 0
        sliceShift = 0
        maxMario = levelWidth - sl
        for i in range(0, maxMario, slice_its_var.get()):
            # Abort when full slice length cannot be reached
            if i + sliceShift >= maxMario:
                break
            sliceCounter += 1

            # Shifts Slice by an amount of empty only tokens on the left
            for length in range(i+sliceShift+1, levelWidth+1):
                firstRow = 0
                for h in range(levelHeight):
                    if level_obj.ascii_level[h][length-1:length] == '-':
                        firstRow += 1
                if firstRow == levelHeight:
                    sliceShift += 1
                else:
                    break

            # Place Mario token for slice
            level_obj.ascii_level, mariocoord = place_token_with_limits(level_obj.ascii_level,
                                                            i + sliceShift, i + sl - 1 + sliceShift, 'M')
            # Add shift if Mario token could not be placed at current slice beginning
            if mariocoord[1] > (i + sliceShift):
                sliceShift += mariocoord[1] - (i + sliceShift)

            # Place Finish token for slice
            level_obj.ascii_level, flagcoord = place_token_with_limits(level_obj.ascii_level,
                                                            i + sliceShift, i + sl - 1 + sliceShift, 'F')
            slices.append((mariocoord, flagcoord))  # Safe slice borders

            if flagcoord[1]-mariocoord[1]+1 != slice_length_var.get():
                s = "Current length %s is not full slice length %s", \
                    str(flagcoord[1]-mariocoord[1]+1), slice_length_var.get()
                error_msg.set(s)
                print(s)
                print(level_obj.ascii_level)

            slicenumber = "%03d" % sliceCounter
            is_loaded.set(False)
            for ai in ai_sel_list:
                # Play level with agent
                threads.append(
                    spawn_thread(q, play_level, ai, False, standard_agent_time, False, True, None, slicenumber))
            for thread in threads:
                thread.join()
            delete_mario_finish()
            da_progressbar['value'] += 100 / ((levelWidth - sl) / 4)
        da_progressbar['value'] = 100
        # Reset to original Mario and Finish tokens
        set_mario_finish(marfin)
        redraw_image()

        # read results from file
        results = []
        with FileLock(GAME_RESULT_PATH + LOCK_EXT):
            with open(GAME_RESULT_PATH, 'r') as file:
                for line in file:
                    results.append(json.loads(line))
        error_msg.set("Iterating: Results collected")

        # Results into dataframe
        # column descriptions for pandas dataframe
        result_columns = ["file_name", "agent", "completion_percentage", "time_needed", "total_time"]
        # Appends "completion_factor", "ai_strength", "difficulty_score"
        result_dataframe = pd.DataFrame(results, columns=result_columns)

        # Get difficulty per result and average over slice
        slice_difficulty_df = get_difficulty(result_dataframe)
        error_msg.set("Iterating: difficulty on slices calculated")

        # Print in console for overview
        # print_results(results)

        # Get level difficulty by slice average
        level_difficulty = slice_difficulty_df[['difficulty_score']].mean()[0]
        slice_difficulty_df = pd.concat([slice_difficulty_df,
                                         pd.DataFrame([{'difficulty_score': level_difficulty}],
                                                      index = ['Full'], columns = slice_difficulty_df.columns)])
        current_difficulty_value.set(round(level_difficulty, 3))

        fig = slice_difficulty_df.plot(x=slice_difficulty_df.index.name, xlabel="Slice",
                                       y="difficulty_score", ylabel="Difficulty", kind="bar").get_figure()
        fig.savefig(OUT+"Difficulty_Plot.png")

        error_msg.set("Iterating Finished")
        use_gen.set(remGen)
        is_loaded.set(True)

        return slice_difficulty_df, slices

    def save_slice(bounds, folder=OUT, name="slice"):
        slice_ascii = []
        if not os.path.exists(folder):
            os.makedirs(folder)

        for row in level_obj.ascii_level:
            slice_ascii.append(row[bounds[0]:bounds[1]+1])
        slice_file = open(os.path.join(folder, name+".txt"), 'w')
        slice_file.write("\n".join(slice_ascii))
        slice_file.close()

    def diff_slice_folder():
        d = fd.askdirectory(initialdir=CURDIR)
        levels = [os.path.join(d, f) for f in os.listdir(d) if f[-3:] == "txt"]
        for l in levels:
            load_level_by_path(l)
            dataframe, slice_boundaries = ai_iterate_level("results")
            marfin = delete_mario_finish()
            for idx in dataframe.index:
                if idx.isnumeric():
                    # Gets x boundaries of slice
                    bounds = slice_boundaries[int(idx)-1][0][1], slice_boundaries[int(idx)-1][1][1]
                    # Safe folder for slice of difficulty: 0.2521 -> 025/ ; 1.7986 -> 179/
                    diff_folder = "%03d/" % math.floor(dataframe['difficulty_score'].loc[idx] * 100)
                    save_slice(bounds, os.path.join(OUT, diff_folder), level_obj.name + "_" + idx)
            set_mario_finish(marfin)

    def ascii_to_image(filepath=None):
        if filepath:
            folder, name = os.path.split(filepath)

            lev, tok = read_level_from_file(folder, name)
            slice_ascii = one_hot_to_ascii_level(lev, tok)

            ImgGen.render(slice_ascii).save(filepath.replace(".txt", ".png"))
        return len(slice_ascii[0])-1, len(slice_ascii)

    def wfc_run(file_name, ascii_file, length, height, outputfolder=OUT):
        wfc_control.execute_wfc(filename=file_name, pattern_width=5, output_size=[length, height],
                                output_periodic=False, input_periodic=False, logging=True,
                                output_destination=outputfolder, ground=-1, rotations=1,
                                mario_version=True, ascii_file=ascii_file)

    def try_once():
        folder, name = os.path.split(r"C:\Studium\Bachelorarbeit_save\DDA_PCG_TOAD_GAN\TOAD-GUI\levels\original_diff_slice_v1\013\lvl_1-1_004.txt")
        lev, tok = read_level_from_file(folder, name)
        slice_ascii = one_hot_to_ascii_level(lev, tok)
        length, height = len(slice_ascii[0]) - 1, len(slice_ascii)

        genpath = os.path.join(folder, "gen/")
        if not os.path.exists(genpath):
            os.makedirs(genpath)
        spawn_thread(q, wfc_run, name[:-4], slice_ascii, length, height, genpath).join()

    def wfc_recreate():
        da_progressbar['value'] = 0
        directory = fd.askdirectory(initialdir=CURDIR)
        subdirs = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
        sdlen = len(subdirs)
        print(directory)
        print(subdirs)
        for d in subdirs:
            print(d)
            levels = [os.path.join(d, f) for f in os.listdir(d) if f[-3:] == "txt"]
            llen = len(levels)
            for l in levels:
                print(l)
                slice_path = l
                folder, name = os.path.split(slice_path)
                lev, tok = read_level_from_file(folder, name)
                slice_ascii = one_hot_to_ascii_level(lev, tok)
                length, height = len(slice_ascii[0])-1, len(slice_ascii)

                genpath = os.path.join(folder, "gen/")
                if not os.path.exists(genpath):
                    os.makedirs(genpath)
                spawn_thread(q, wfc_run, slice_ascii, length, height, genpath).join()
                da_progressbar['value'] += (100 / llen) * (1 / sdlen)
        da_progressbar['value'] = 100
    # ---------------------------------------- Layout ----------------------------------------

    settings = ttk.Frame(root, padding=(15, 15, 15, 15), width=1000, height=1000)  # Main Frame

    banner = ttk.Label(settings, image=banner_icon)
    welcome_message = ttk.Label(settings, text="Welcome to TOAD-GUI, a Framework to view and playtest Mario levels "
                                               "generated by TOAD-GAN.", width=95, anchor='center')

    # Displays loaded path
    fpath_label = ttk.Label(settings, textvariable=load_string_gen, width=100)

    # Top Buttons
    load_lev_button = ttk.Button(settings, compound='top', image=load_level_icon, width=35,
                                 text='Open Level', command=lambda: spawn_thread(q, load_level))
    load_gen_button = ttk.Button(settings, compound='top', image=load_generator_icon, width=35,
                                 text='Open Generator', command=lambda: spawn_thread(q, load_generator))
    save_button = ttk.Button(settings, compound='top', image=save_level_icon,
                             text='Save Level/Image', state='disabled', command=lambda: spawn_thread(q, save_txt))
    gen_button = ttk.Button(settings, compound='top', image=generate_level_icon,
                            text='Generate level', state='disabled', command=lambda: spawn_thread(q, generate))

    # Size Entries
    size_frame = ttk.Frame(settings, padding=(1, 1, 1, 1))
    h_label = ttk.Label(size_frame, text="Size:")
    h_entry = ttk.Entry(size_frame, textvariable=level_h, validate="key", width=3, justify='right', state='disabled')
    l_label = ttk.Label(size_frame, text="X")
    l_entry = ttk.Entry(size_frame, textvariable=level_l, validate="key", width=3, justify='right', state='disabled')

    vcmd_h = (h_entry.register(on_validate), '%P', '%d')
    vcmd_l = (l_entry.register(on_validate), '%P', '%d')

    h_entry.configure(validatecommand=vcmd_h)
    l_entry.configure(validatecommand=vcmd_l)

    # Callback to set buttons active/inactive depending on load state
    def set_button_state(t1, t2, t3):
        if use_gen.get():
            gen_button.state(['!disabled'])
            if not is_loaded.get():
                save_button.state(['disabled'])
            else:
                save_button.state(['!disabled'])
            h_entry.state(['!disabled'])
            l_entry.state(['!disabled'])
            p_c_tabs.tab(2, state="normal")
        else:
            gen_button.state(['disabled'])
            if not is_loaded.get():
                save_button.state(['disabled'])
            else:
                save_button.state(['!disabled'])
                editmode.set(False)
            h_entry.state(['disabled'])
            l_entry.state(['disabled'])
            p_c_tabs.tab(2, state="disabled")
        return

    use_gen.trace("w", callback=set_button_state)

    p_c_tabs = ttk.Notebook(settings)
    # Play and Controls Notebook Tab
    p_c_frame = ttk.Frame(settings)
    p_c_tabs.add(p_c_frame, text="Play/Controls")
    play_button = ttk.Button(p_c_frame, compound='top', image=play_level_icon,
                             text='Play level', state='disabled',
                             command=lambda: spawn_thread(q, play_level, "Human", False, 180))
    play_ai_button = ttk.Button(p_c_frame, compound='top', image=play_ai_level_icon,
                                text='AI Play level', state='disabled',
                                command=lambda: spawn_thread(q, play_level, ai_options_combobox.get(), False, 30))

    selected_ai = StringVar()
    base_ais = (
        'robinBaumgarten', 'andySloane', 'doNothing', 'glennHartmann', 'michal', 'random', 'sergeyKarakovskiy',
        'sergeyPolikarpov', 'spencerSchumann', 'trondEllingsen')
    advanced_ais = (
        'astar', 'astarDistanceMetric', 'astarFast', 'astarJump', 'astarPlanning',
        'astarPlanningDynamic', 'astarWindow', 'robinBaumgartenSlim', 'robinBaumgartenSlimImproved',
        'robinBaumgartenSlimWindowAdvance')
    # Currently not working , 'astarGrid', 'astarWaypoints'

    ai_sel_list = ["astar", "astarPlanningDynamic", "robinBaumgarten", "random"]

    ai_options_combobox = ttk.Combobox(p_c_frame, textvariable=selected_ai)
    ai_options_combobox['values'] = ai_sel_list
    ai_options_combobox['state'] = 'readonly'
    ai_options_combobox.current(0)

    use_selected_ai = BooleanVar()
    use_selected_ai.set(True)

    def ai_switch():
        if use_selected_ai.get():
            ai_options_combobox['values'] = ai_sel_list
        else:
            ai_options_combobox['values'] = base_ais + advanced_ais
        ai_options_combobox.current(0)

    use_selected_ai_checkbutton = ttk.Checkbutton(p_c_frame,
                                                  text='Use selected AI',
                                                  command=ai_switch,
                                                  variable=use_selected_ai)
    # Difficulty Adjustment Notebook Tab
    difficulty_frame = ttk.Frame(settings)
    p_c_tabs.add(difficulty_frame, text="Difficulty Adjustment")
    iterate_button = ttk.Button(difficulty_frame, compound='top', image=iterate_level_icon,
                                text='Determine difficulty with ai', state='disabled',
                                command=lambda: spawn_thread(q, ai_iterate_level))
    slice_its_var = IntVar()
    slice_its_var.set(4)
    slice_its_label = ttk.Label(difficulty_frame, text="Slice Iterations: ")
    slice_its_entry = ttk.Entry(difficulty_frame, textvariable=slice_its_var, validate="key", width=3)

    slice_length_var = IntVar()
    slice_length_var.set(24)
    slice_length_label = ttk.Label(difficulty_frame, text="Slice Length: ")
    slice_length_entry = ttk.Entry(difficulty_frame, textvariable=slice_length_var, validate="key", width=3)

    current_difficulty_value = DoubleVar()
    current_difficulty_label = ttk.Label(difficulty_frame, text="Current Difficulty: Not determined")

    def difficulty_value_changed(t1, t2, t3):
        current_difficulty_label.config(text="Current Difficulty: " + str(current_difficulty_value.get()))

    current_difficulty_value.trace("w", callback=difficulty_value_changed)

    das_value = DoubleVar()
    das_value_entry = ttk.Entry(difficulty_frame, textvariable=das_value, width=4)

    def d_slider_changed(event):
        das_value.set(round(das_value.get(), 2))

    das_label = ttk.Label(difficulty_frame, text="Difficulty Adjustment")
    difficulty_adjustment_slider = ttk.Scale(difficulty_frame, from_=0.000, to=1.000, orient='horizontal', variable=das_value,
                                             command=d_slider_changed, length=250)
    das_value.set(0.000)

    da_progressbar = tkinter.ttk.Progressbar(difficulty_frame, orient='horizontal', mode='determinate', length=800)

    slice_extract_button = ttk.Button(difficulty_frame, compound='top', image=extract_slices_icon,
                                text='Extract Slices', command=lambda: spawn_thread(q, diff_slice_folder))

    wfc_sample_button = ttk.Button(difficulty_frame, compound='top',  # image=extract_slices_icon,
                                text='WFC Resample', command=lambda: spawn_thread(q, try_once))

    edit_tab = ttk.Frame(settings)
    def on_tab_change(event):
        tab = event.widget.tab('current')['text']
        if tab == 'Edit Mode':
            editmode.set(True)

    p_c_tabs.bind('<<NotebookTabChanged>>', on_tab_change)
    # Level Preview image
    image_label = ScrollableImage(settings, image=levelimage, height=271)

    # Token edit function
    def change_token(tok, x, y):
        level_obj.oh_level[0, :, y, x] = 0
        level_obj.oh_level[0, level_obj.tokens.index(tok), y, x] = 1
        level_obj.ascii_level = one_hot_to_ascii_level(level_obj.oh_level, level_obj.tokens)
        toggle_editmode(None, None, None)

    # Popup Menu for the token edit function
    def popup_edit(event):
        e_menu = Menu(root, tearoff=0)
        # Get actual x,y position from click position
        tok_x = int(event.widget.canvasx(event.x) / 16)
        tok_y = int(event.widget.canvasy(event.y) / 16)

        if is_loaded.get():
            try:
                for i, t in enumerate(level_obj.tokens):
                    if not (t == 'M' or t == 'F'):  # Mario or Flag can not be placed, but are placed automatically
                        e_menu.add_command(image=token_img_dict[t], label=t, compound='left',
                                           command=lambda tok=t: change_token(tok, tok_x, tok_y))
            except TypeError:
                e_menu.add_command(label="No changeable Level loaded")
        else:
            e_menu.add_command(label="No changeable Level loaded")

        e_menu.tk_popup(event.x_root, event.y_root)

    # Bind right-click to pop-up menu
    if platform.system() == 'Darwin':
        image_label.bind("<Button-2>", popup_edit)  # MacOS has Button 2 for some reason
    else:
        image_label.bind("<Button-3>", popup_edit)

    # Enable/Disable Play Button when loaded/unloaded
    def set_play_state(t1, t2, t3):
        if is_loaded.get():
            play_button.state(['!disabled'])
            play_ai_button.state(['!disabled'])
            iterate_button.state(['!disabled'])

            image_label.change_image(level_obj.image)
        else:
            play_button.state(['disabled'])
            play_ai_button.state(['disabled'])
            iterate_button.state(['disabled'])
        toggle_editmode(t1, t2, t3)
        return

    is_loaded.trace("w", callback=set_play_state)

    # Frame displaying the Game's controls
    controls_frame = ttk.LabelFrame(p_c_frame, padding=(5, 5, 5, 5), text='Controls')
    contr_a = ttk.Label(controls_frame, text=' a :')
    contr_s = ttk.Label(controls_frame, text=' s :')
    contr_l = ttk.Label(controls_frame, text='<- :')
    contr_r = ttk.Label(controls_frame, text='-> :')

    descr_a = ttk.Label(controls_frame, text='Sprint/Throw Fireball')
    descr_s = ttk.Label(controls_frame, text='Jump')
    descr_l = ttk.Label(controls_frame, text='Move left')
    descr_r = ttk.Label(controls_frame, text='Move right')

    error_label = ttk.Label(settings, textvariable=error_msg)

    # ---------------------------------------- Grid Layout ----------------------------------------

    # On root:
    settings.grid(column=0, row=0, sticky=(N, S, E, W))

    # On settings:
    banner.grid(column=1, row=0, columnspan=2, sticky=(N, S))
    welcome_message.grid(column=1, row=1, columnspan=2, sticky=(N, S), padx=5, pady=8)
    load_lev_button.grid(column=2, row=3, sticky=(N, S, E, W), padx=5, pady=5)
    load_gen_button.grid(column=1, row=3, sticky=(N, S, E, W), padx=5, pady=5)
    gen_button.grid(column=1, row=4, sticky=(N, S, E, W), padx=5, pady=5)
    save_button.grid(column=2, row=4, sticky=(N, S, E, W), padx=5, pady=5)
    image_label.grid(column=0, row=6, columnspan=4, sticky=(N, E, W), padx=5, pady=8)
    p_c_tabs.grid(column=1, row=7, columnspan=2, sticky=(N, S, E, W), padx=5, pady=5)
    fpath_label.grid(column=0, row=99, columnspan=4, sticky=(S, E, W), padx=5, pady=5)
    error_label.grid(column=0, row=100, columnspan=4, sticky=(S, E, W), padx=5, pady=1)
    size_frame.grid(column=1, row=5, columnspan=1, sticky=(N, S), padx=5, pady=2)

    # On size_frame:
    h_label.grid(column=0, row=0, sticky=(N, S, E), padx=1, pady=0)
    l_entry.grid(column=1, row=0, sticky=(N, S), padx=1, pady=0)
    l_label.grid(column=2, row=0, sticky=(N, S), padx=1, pady=0)
    h_entry.grid(column=3, row=0, sticky=(N, S), padx=1, pady=0)

    # On p_c_frame:
    play_button.grid(column=0, row=0, rowspan=3, sticky=(N, S, E, W), padx=5, pady=5)
    play_ai_button.grid(column=1, row=0, rowspan=3, sticky=(N, S, E, W), padx=5, pady=5)
    controls_frame.grid(column=2, row=0, sticky=(N, S, E, W), padx=5, pady=5)
    use_selected_ai_checkbutton.grid(column=2, row=1, sticky=(N, S, E, W), padx=5, pady=5)
    ai_options_combobox.grid(column=2, row=2, columnspan=2, sticky=(N, S, E, W), padx=5, pady=5)

    # On controls_frame
    contr_a.grid(column=0, row=0, sticky=(N, S, E), padx=1, pady=1)
    contr_s.grid(column=0, row=1, sticky=(N, S, E), padx=1, pady=1)
    contr_l.grid(column=0, row=2, sticky=(N, S, E), padx=1, pady=1)
    contr_r.grid(column=0, row=3, sticky=(N, S, E), padx=1, pady=1)
    descr_a.grid(column=1, row=0, sticky=(N, S, W), padx=1, pady=1)
    descr_s.grid(column=1, row=1, sticky=(N, S, W), padx=1, pady=1)
    descr_l.grid(column=1, row=2, sticky=(N, S, W), padx=1, pady=1)
    descr_r.grid(column=1, row=3, sticky=(N, S, W), padx=1, pady=1)

    # On difficulty_frame
    iterate_button.grid(column=0, row=0, sticky=(N, S, E, W), padx=5, pady=5)
    slice_its_label.grid(column=2, row=0, sticky=(N, W), padx=5, pady=5)
    slice_its_entry.grid(column=3, row=0, sticky=(N), padx=5, pady=5)
    slice_length_label.grid(column=2, row=0, sticky=(W), padx=5, pady=5)
    slice_length_entry.grid(column=3, row=0, padx=5, pady=5)
    current_difficulty_label.grid(column=0, row=1, columnspan=4, sticky=(N), padx=5, pady=5)
    das_label.grid(column=4, row=0, sticky=(N), padx=5, pady=5)
    das_value_entry.grid(column=5, row=0, sticky=(N), padx=5, pady=5)
    difficulty_adjustment_slider.grid(column=4, row=0, columnspan=3, pady=5)
    da_progressbar.grid(column=0, row=1, columnspan=6, sticky=(S), padx=5, pady=5)
    slice_extract_button.grid(column=5, row=1, sticky=(N))
    wfc_sample_button.grid(column=4, row=1, sticky=(N))

    # Column/Rowconfigure
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    settings.columnconfigure(0, weight=1)
    settings.columnconfigure(1, weight=1)
    settings.columnconfigure(2, weight=1)
    settings.columnconfigure(3, weight=1)
    settings.rowconfigure(0, weight=1)
    settings.rowconfigure(1, weight=1)
    settings.rowconfigure(2, weight=1)
    settings.rowconfigure(3, weight=1)
    settings.rowconfigure(4, weight=1)
    settings.rowconfigure(5, weight=1)
    settings.rowconfigure(6, weight=2)
    settings.rowconfigure(7, weight=1)
    settings.rowconfigure(8, weight=1)
    settings.rowconfigure(99, weight=1)
    settings.rowconfigure(100, weight=1)

    p_c_frame.columnconfigure(0, weight=2)
    p_c_frame.columnconfigure(1, weight=2)
    p_c_frame.columnconfigure(2, weight=2)

    controls_frame.columnconfigure(0, weight=1)
    controls_frame.columnconfigure(1, weight=1)
    controls_frame.rowconfigure(0, weight=1)
    controls_frame.rowconfigure(1, weight=1)
    controls_frame.rowconfigure(2, weight=1)
    controls_frame.rowconfigure(3, weight=1)

    difficulty_frame.columnconfigure(0, weight=3)
    difficulty_frame.columnconfigure(1, weight=1)
    difficulty_frame.columnconfigure(2, weight=1)
    difficulty_frame.columnconfigure(3, weight=1)
    difficulty_frame.columnconfigure(4, weight=1)
    difficulty_frame.columnconfigure(5, weight=1)
    difficulty_frame.columnconfigure(6, weight=1)
    difficulty_frame.rowconfigure(0, weight=1)
    difficulty_frame.rowconfigure(1, weight=1)
    # ---------------------------------------- Edit Mode ----------------------------------------

    # Define Variables
    editmode = BooleanVar()
    bbox_x1 = IntVar()
    bbox_x2 = IntVar()
    bbox_y1 = IntVar()
    bbox_y2 = IntVar()
    edit_scale = IntVar()
    scale_info = StringVar()

    # Set values
    editmode.set(False)
    bbox_x1.set(0)
    bbox_x2.set(16)
    bbox_y1.set(0)
    bbox_y2.set(16)
    edit_scale.set(0)
    scale_info.set("Scale 0 window: 8x8")

    # Placeholder for the noise representation
    noise_holder = Image.new('RGB', (8, 8), (255, 255, 255))
    noiseimage = ImageTk.PhotoImage(noise_holder)

    # ---------------------------------------- Edit Mode Widgets ----------------------------------------

    # Set Edit mode Tab
    p_c_tabs.add(edit_tab, text="Edit Mode", state='disabled')
    em_tooltip = Tooltip(edit_tab,
                         text="Right click the image to change a token directly. \n"
                              "Edit mode allows for resampling parts of a generated level.",
                         wraplength=250, bg="white", enabled=True, waittime=100)

    # Edit mode frame
    emode_frame = ttk.LabelFrame(edit_tab, text="Edit mode controls", padding=(5, 5, 5, 5))
    # Bounding Box frame
    bbox_frame = ttk.LabelFrame(emode_frame, text="Bounding Box", padding=(5, 5, 5, 5))

    # Bounding box Entries
    x1_label = ttk.Label(bbox_frame, text="x1:", width=5, anchor="e")
    x1_entry = ttk.Entry(bbox_frame, textvariable=bbox_y1, validate="key", width=3, justify='right')
    x2_label = ttk.Label(bbox_frame, text="x2:")
    x2_entry = ttk.Entry(bbox_frame, textvariable=bbox_y2, validate="key", width=3, justify='right')
    y1_label = ttk.Label(bbox_frame, text=" y1:", width=5, anchor="e")
    y1_entry = ttk.Entry(bbox_frame, textvariable=bbox_x1, validate="key", width=3, justify='right')
    y2_label = ttk.Label(bbox_frame, text=" y2:")
    y2_entry = ttk.Entry(bbox_frame, textvariable=bbox_x2, validate="key", width=3, justify='right')

    bbox_tt_string = "Controls the red bounding box. The yellow box shows which tokens are also " \
                     "influenced by a change in this scale."

    x1_tooltip = Tooltip(x1_label, text=bbox_tt_string, wraplength=250, bg="white", enabled=False)
    x2_tooltip = Tooltip(x2_label, text=bbox_tt_string, wraplength=250, bg="white", enabled=False)
    y1_tooltip = Tooltip(y1_label, text=bbox_tt_string, wraplength=250, bg="white", enabled=False)
    y2_tooltip = Tooltip(y2_label, text=bbox_tt_string, wraplength=250, bg="white", enabled=False)

    # Difficulty Adjustment TT
    slice_its_tt_string = "Lower number results in more shifts and therefore longer runtime"

    slice_its_tt = Tooltip(slice_its_label, text=slice_its_tt_string, wraplength=250, bg="white")

    # Scale entry and
    sc_frame = ttk.Frame(emode_frame, padding=(0, 5, 0, 5))
    sc_label = ttk.Label(sc_frame, text="Scale:")
    sc_entry = ttk.Entry(sc_frame, textvariable=edit_scale, validate="key", width=3, justify='right')

    # Noise image and info label
    sc_noise_image = ScrollableImage(emode_frame, image=noiseimage, height=50, width=300)
    noise_tooltip = Tooltip(sc_noise_image,
                            text="This is a visualization of the noise map in the chosen scale. "
                                 "Pixels to be resampled are marked red.",
                            wraplength=250, bg="white", enabled=False)

    sc_info_label = ttk.Label(sc_frame, textvariable=scale_info)
    sc_tooltip = Tooltip(sc_info_label,
                         text="Noise influence is a learned parameter that indicates "
                              "how much influence a change of the noise map will have on this scale.",
                         wraplength=250, bg="white", enabled=False)

    # Entry Validation
    vcmd_x1 = (x1_entry.register(on_validate), '%P', '%d')
    vcmd_x2 = (x2_entry.register(on_validate), '%P', '%d')
    vcmd_y1 = (y1_entry.register(on_validate), '%P', '%d')
    vcmd_y2 = (y2_entry.register(on_validate), '%P', '%d')
    vcmd_sc = (sc_entry.register(on_validate), '%P', '%d')

    x1_entry.configure(validatecommand=vcmd_x1)
    x2_entry.configure(validatecommand=vcmd_x2)
    y1_entry.configure(validatecommand=vcmd_y1)
    y2_entry.configure(validatecommand=vcmd_y2)
    sc_entry.configure(validatecommand=vcmd_sc)

    # Resample Button and info
    resample_button = ttk.Button(emode_frame, text="Resample", state='disabled',
                                 command=lambda: spawn_thread(q, re_sample))
    sample_info = ttk.Label(emode_frame, text="Right click to edit Tokens directly.\n"
                                              "Resampling will regenerate the level,\n"
                                              "so prior Token edits will be lost.")

    # TOAD-GAN resample function
    def re_sample():
        if not level_obj.scales:
            error_msg.set("Can't resample loaded level.")  # should not be reachable
        else:
            is_loaded.set(False)
            # Get scaled Bounding Box
            sc = level_obj.scales[edit_scale.get()].shape[-1] / level_obj.oh_level.shape[-1]
            scaled_bbox = (round(bbox_x1.get() * sc), round(bbox_x2.get() * sc),
                           round(bbox_y1.get() * sc), round(bbox_y2.get() * sc))

            new_states = [level_obj.scales, level_obj.noises]

            # Get x,y scales from height and length
            sc_h = level_h.get() / toadgan_obj.reals[-1].shape[-2]
            sc_l = level_l.get() / toadgan_obj.reals[-1].shape[-1]

            # Regenerate level from noise (this will reset prior edits)
            level, scales, noises = generate_sample(toadgan_obj.Gs, toadgan_obj.Zs, toadgan_obj.reals,
                                                    toadgan_obj.NoiseAmp, toadgan_obj.num_layers,
                                                    toadgan_obj.token_list, in_states=new_states,
                                                    gen_start_scale=edit_scale.get(), is_bboxed=True,
                                                    bbox=scaled_bbox, scale_v=sc_h, scale_h=sc_l)
            level_obj.oh_level = level.cpu()
            level_obj.scales = scales
            level_obj.noises = noises

            level_obj.ascii_level = one_hot_to_ascii_level(level, toadgan_obj.token_list)
            redraw_image(scale=edit_scale.get())

            is_loaded.set(True)

    # Grid/Remove Edit mode widgets
    def toggle_editmode(t1, t2, t3):
        if editmode.get():
            # On edit_tab:
            emode_frame.grid(column=0, row=0, rowspan=3, sticky=(E, W), padx=10)

            # On emode_frame:
            bbox_frame.grid(column=0, row=0, columnspan=1, sticky=(E, W), padx=5, pady=5)
            resample_button.grid(column=0, row=5, columnspan=3, sticky=(N, S, E, W), padx=5, pady=5)
            sample_info.grid(column=0, row=4, columnspan=3, sticky=(N, S), padx=5, pady=5)
            sc_frame.grid(column=1, row=0, columnspan=3, sticky=(N, S, E, W), padx=5, pady=5)
            sc_label.grid(column=0, row=0, columnspan=1, sticky=(N, S, E), padx=1, pady=5)
            sc_entry.grid(column=1, row=0, columnspan=1, sticky=(N, S, W), padx=1, pady=5)
            sc_info_label.grid(column=0, row=1, columnspan=2, sticky=(N, S, E, W), padx=1, pady=5)
            sc_noise_image.grid(column=0, row=2, columnspan=2, sticky=(N, S), padx=1, pady=5)

            # On bbox_frame:
            x1_label.grid(column=0, row=0, sticky=(N, S, E), padx=1, pady=1)
            x1_entry.grid(column=1, row=0, sticky=(N, S, W), padx=5, pady=1)
            x2_label.grid(column=0, row=1, sticky=(N, S, E), padx=1, pady=1)
            x2_entry.grid(column=1, row=1, sticky=(N, S, W), padx=5, pady=1)
            y1_label.grid(column=2, row=0, sticky=(N, S, E), padx=1, pady=1)
            y1_entry.grid(column=3, row=0, sticky=(N, S, W), padx=5, pady=1)
            y2_label.grid(column=2, row=1, sticky=(N, S, E), padx=1, pady=1)
            y2_entry.grid(column=3, row=1, sticky=(N, S, W), padx=5, pady=1)

            # Column/Rowconfigure
            emode_frame.columnconfigure(0, weight=0)
            emode_frame.columnconfigure(1, weight=1)
            emode_frame.rowconfigure(0, weight=1)
            emode_frame.rowconfigure(1, weight=1)
            emode_frame.rowconfigure(2, weight=1)
            emode_frame.rowconfigure(3, weight=1)
            emode_frame.rowconfigure(4, weight=1)
            emode_frame.rowconfigure(5, weight=1)

            bbox_frame.columnconfigure(0, weight=1)
            bbox_frame.columnconfigure(1, weight=1)
            bbox_frame.columnconfigure(2, weight=1)
            bbox_frame.columnconfigure(3, weight=1)
            bbox_frame.rowconfigure(0, weight=1)
            bbox_frame.rowconfigure(1, weight=1)

            sc_frame.columnconfigure(0, weight=1)
            sc_frame.columnconfigure(1, weight=1)
            sc_frame.rowconfigure(0, weight=1)
            sc_frame.rowconfigure(1, weight=1)
            sc_frame.rowconfigure(2, weight=1)

            redraw_image(True, rectangle=[(bbox_y1.get() * 16, bbox_x1.get() * 16),
                                          (bbox_y2.get() * 16, bbox_x2.get() * 16)], scale=edit_scale.get())

            update_scale_info(t1, t2, t3)

        else:
            emode_frame.grid_forget()
            redraw_image(False)

    editmode.trace("w", callback=toggle_editmode)

    # Update scale info label, noise image and resample button availability
    def update_scale_info(t1, t2, t3):
        try:
            # Correct bbox values, TODO: BBox entries are not user friendly - lower value can't be set first
            if bbox_y1.get() < 0:
                bbox_y1.set(0)
            if bbox_x1.get() < 0:
                bbox_x1.set(0)
            if bbox_y2.get() >= len(level_obj.ascii_level[-1]):
                bbox_y2.set(len(level_obj.ascii_level[-1]))
            if bbox_x2.get() >= len(level_obj.ascii_level):
                bbox_x2.set(len(level_obj.ascii_level))
            if bbox_y1.get() >= bbox_y2.get():
                bbox_y1.set(bbox_y2.get() - 1)
            if bbox_x1.get() >= bbox_x2.get():
                bbox_x1.set(bbox_x2.get() - 1)

            redraw_image(True, rectangle=[(bbox_y1.get() * 16, bbox_x1.get() * 16),
                                          (bbox_y2.get() * 16, bbox_x2.get() * 16)], scale=edit_scale.get())

            # Show Noise Amplitude, which indicates how much influence this noise change will have
            scale_info.set("Noise influence: %.4f\n"
                           % (toadgan_obj.NoiseAmp[edit_scale.get()]))
            sc_tooltip.enabled = True

            # Calculate scaled bounding box
            sc = level_obj.scales[edit_scale.get()].shape[-1] / level_obj.oh_level.shape[-1]
            scaled_bbox = [(round(bbox_y1.get() * sc), round(bbox_x1.get() * sc)),
                           (round(bbox_y2.get() * sc), round(bbox_x2.get() * sc))]

            # Get first tokens noise map on current scale as representation
            noise_holder = Image.fromarray(level_obj.noises[edit_scale.get()][0, 0, 3:-3, 3:-3].cpu().numpy() * 255)
            noise_holder = noise_holder.convert('RGB')

            # Draw adjusted bounding box on noise map
            if not (round(bbox_y1.get() * sc) == round(bbox_y2.get() * sc) or
                    round(bbox_x1.get() * sc) == round(bbox_x2.get() * sc)):
                n_draw = ImageDraw.Draw(noise_holder, 'RGBA')
                rectanglebox = scaled_bbox
                rectanglebox[1] = (max(round(bbox_y1.get() * sc), round(bbox_y2.get() * sc) - 1),
                                   max(round(bbox_x1.get() * sc), round(bbox_x2.get() * sc) - 1))
                n_draw.rectangle(rectanglebox, fill=(255, 0, 0, 128))

            # Make image bigger, so you can see it better
            noise_holder = noise_holder.resize((noise_holder.size[0] * 2, noise_holder.size[1] * 2), Image.NEAREST)

            # Set noise image
            noiseimage = ImageTk.PhotoImage(noise_holder)
            sc_noise_image.change_image(noiseimage)

            resample_button.state(['!disabled'])  # Allow resampling
            noise_tooltip.enabled = True
            x1_tooltip.enabled = True
            x2_tooltip.enabled = True
            y1_tooltip.enabled = True
            y2_tooltip.enabled = True

        except TclError:
            scale_info.set("No scale")
            resample_button.state(['disabled'])
            sc_tooltip.enabled = False
            noise_tooltip.enabled = True
            x1_tooltip.enabled = True
            x2_tooltip.enabled = True
            y1_tooltip.enabled = True
            y2_tooltip.enabled = True

        except IndexError:
            scale_info.set("Scale out of range")
            resample_button.state(['disabled'])
            sc_tooltip.enabled = False
            noise_tooltip.enabled = True
            x1_tooltip.enabled = True
            x2_tooltip.enabled = True
            y1_tooltip.enabled = True
            y2_tooltip.enabled = True

        except TypeError:
            scale_info.set("No editable level loaded")
            noise_holder = Image.new('RGB', (8, 8), (255, 255, 255))
            noiseimage = ImageTk.PhotoImage(noise_holder)
            sc_noise_image.change_image(noiseimage)
            resample_button.state(['disabled'])
            sc_tooltip.enabled = False
            noise_tooltip.enabled = False
            x1_tooltip.enabled = False
            x2_tooltip.enabled = False
            y1_tooltip.enabled = False
            y2_tooltip.enabled = False

    edit_scale.trace("w", callback=update_scale_info)
    bbox_x1.trace("w", callback=update_scale_info)
    bbox_x2.trace("w", callback=update_scale_info)
    bbox_y1.trace("w", callback=update_scale_info)
    bbox_y2.trace("w", callback=update_scale_info)

    # Move scrollbar to middle (to have message on the test image appear centered)
    root.update()
    root.after(1, image_label.move_scrollbar_to_middle)

    # Run Window
    root.mainloop()
