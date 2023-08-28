import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

TMP_PATH = "TOAD-GUI/tmp"
EVAL_PATH = "TOAD-GUI/evaluation"
DIFF_SLICE_PATH = "TOAD-GUI/levels/difficulty_slices"
ALT_DIFF_SLICE_PATH = "TOAD-GUI/levels/6-3_difficulty_slices"
goal_list = ["G013", "G015", "G017", "G019", "G022", "G025", "G030", "G040", "G050", "G060", "G108"]

generators = {f: os.path.join(EVAL_PATH, f) for f in os.listdir(EVAL_PATH) if "TOAD_GAN" in f and os.path.isdir(os.path.join(EVAL_PATH, f))}
gen_goals = {} # Generator paired with each GOAL and it's initial difficulty
gen_diff = {} # Generator paired with only the initial difficulties of the goals
goal_adj_diff = []
goal_mean_adjustment= []
gen_goal_duplicates = {}

def extractEvalInformation():
    for gen, path in generators.items():
        gen_goal_duplicates[gen] = {}
        gen_goals[gen] = [f.replace(gen, "") for f in os.listdir(path)]
        gen_diff[gen] = {}
        gen_goal_duplicates[gen] = {g: 0 for g in goal_list}
        # Filtern der Liste und Entfernen von Elementen, die mit "_uncomp" oder "_errorabort" enden
        filtered_list = [g for g in gen_goals[gen] if (not '_uncomp' in g) and (not '_errorabort' in g)]
        # Weitere Filterung, um nur das erste relevante Ergebnis beizubehalten
        for goal in goal_list:
            for item in filtered_list:
                if goal in item:
                    goal_diff_path = os.path.join(path, gen+item)
                    goal_diff_iterations = [os.path.join(goal_diff_path, f) for f in os.listdir(goal_diff_path)
                                            if os.path.isdir(os.path.join(goal_diff_path, f))]
                    init_diff = None
                    end_diff = None
                    pre = None
                    post = None
                    for idx, iteration in enumerate(goal_diff_iterations):
                        for file in os.listdir(iteration):
                            if "pre" in file:
                                pre = float(file.split('_')[1])
                                if idx == 0:
                                    # Get the initial difficulty from the first "pre" image that has it saved in it's filename
                                    init_diff = pre
                                    gen_diff[gen][goal] = pre
                            if "post" in file:
                                post = float(file.split('_')[1])
                                if idx == len(goal_diff_iterations)-1:
                                    # Get difficulty per goal only from valid adjustments
                                    end_diff = post
                        if "dupabort" not in item:
                            goal_mean_adjustment.append([goal, idx, pre, post])
                    if "dupabort" not in item:
                        goal_adj_diff.append([goal, gen, init_diff, end_diff])
                    else:
                        gen_goal_duplicates[gen][goal] += 1

def getGenPlot():
    extractEvalInformation()
    df = pd.DataFrame(columns=goal_list)
    for index, values in gen_diff.items():
        df.loc[index] = [values[column] for column in goal_list]
    print(df.mean(axis=1))
    plt.figure(figsize=(10, 6))
    boxplot = df.T.boxplot()
    boxplot.set_xlabel("Generator")
    boxplot.set_ylabel("Schwierigkeit D")
    plt.xticks(rotation=45)  # Drehen der x-Achsentick-Beschriftungen für bessere Lesbarkeit
    plt.tight_layout()  # Verbessert die Layout-Anordnung der Plots
    plt.savefig(os.path.join(TMP_PATH, 'boxplot_per_generator.png'))

def getDiffPlot():
    extractEvalInformation()
    df = pd.DataFrame(goal_adj_diff, columns=["goal", "generator", "initial", "final"])
    df['adj_deviation'] = abs(df["initial"] - df["final"])
    df['goal_deviation'] = abs(df["goal"].apply(lambda x: float(x[1:])/100) - df["final"])
    result = df.groupby(["goal"]).agg({
        "initial": "mean",
        "final": "mean",
        "adj_deviation": "mean",
        "goal_deviation": "mean",
        "generator": "count"}).reset_index()
    result.rename(columns={"initial": "mean_initial",
                           "final": "mean_final",
                           "adj_deviation": "mean_adj_deviation",
                           "goal_deviation": "mean_goal_deviation",
                           "generator": "datapoints"},
                  inplace=True)
    print(result)
    plt.figure(figsize=(10, 6))
    bar_width = 0.4  # Breite der Säulen
    index = np.arange(len(result["goal"]))

    bars1 = plt.bar(index + bar_width, result["mean_goal_deviation"], width=bar_width, label="Durchschnittliche Abweichung des Endwerts von der Ziel-Schwierigkeitsklasse")
    bars2 = plt.bar(index, result["mean_adj_deviation"], width=bar_width, label="Durchschnittliche Abweichung des Endwerts vom Startwert",
                    alpha=0.7)

    plt.xlabel("Ziel-Schwierigkeitsklasse")
    plt.ylabel("Durchschnittliche Abweichung")
    plt.title("Durchschnittliche Abweichung des Endwerts zum Ziel- und Startwert")
    newlabels = result["goal"].apply(lambda x: x[1:]).str.cat(result["datapoints"].astype(str).apply(lambda x: " (" + x + ")").values.tolist(),
                                       sep='\n')
    plt.xticks(index + bar_width / 2, labels=newlabels)
    plt.legend()
    plt.ylim(0, 0.5)
    plt.yticks(np.arange(0, 0.5, 0.05))
    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > plt.ylim()[1]:
            plt.text(bar.get_x() + bar.get_width() / 2, plt.ylim()[1], f"{height:.2f}", ha="center", va="bottom")
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(TMP_PATH, 'mean_deviation_barplot.png'))

def getDiffAdjustPlot():
    extractEvalInformation()
    df = pd.DataFrame(goal_mean_adjustment, columns=["goal", "iteration", "pre", "post"])
    df["goal2"] = df["goal"].apply(lambda x: float(x[1:])/100)
    df["convergence"] = df.apply(lambda row: abs(row["post"] - row["pre"]) *
                                 (-1 if abs(row["post"]-row["goal2"]) > abs(row["pre"]-row["goal2"]) else 1), axis=1)
    result = df.groupby(["goal"]).agg({"convergence": "mean"}).reset_index()
    result["convergence"] = result["convergence"] * 1000
    plt.figure(figsize=(10, 6))
    bars = plt.bar(result["goal"], result["convergence"], label="Durchschnittliche Näherung zur Ziel-Schwierigkeitsklasse")
    plt.xlabel("Ziel-Schwierigkeitsklasse")
    plt.ylabel("Durchschnittliche Schwierigkeitsannäherung " + r"$10^{-3}$")
    plt.yticks(np.arange(0, 10, 1))
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(TMP_PATH, 'adjustment_convergence.png'))

def getDuplicateAbortPlot():
    extractEvalInformation()
    df = pd.DataFrame(columns=goal_list)
    for index, values in gen_goal_duplicates.items():
        df.loc[index] = [values[column] for column in goal_list]
    plt.figure(figsize=(8.5, 8))

    statframe = df.copy()
    statframe["sum_gen"] = statframe[list(statframe.columns)].sum(axis=1)
    def filter(row, wert):
        return len(row[row == wert].index.tolist())
    abortnum = 3
    statframe['gen_abort'] = df.apply(lambda row: filter(row, abortnum), axis=1)
    statframe.loc["sum_goal"] = statframe[list(statframe.columns)].sum(axis=0)
    statframe.loc["goal_abort"] = df.apply(lambda row: filter(row, abortnum), axis=0)
    print(statframe.astype(pd.Int64Dtype(), errors='ignore'))

    #HEATMAP
    plt.imshow(df, cmap="coolwarm", aspect="auto")
    # Anzeigen der Werte auf den Zellen
    for i in range(len(df)):
        for j in range(len(df.columns)):
            plt.text(j, i, str(df.iloc[i, j]), ha='center', va='center', color='black')
    plt.title("Visualisierung der Duplikate")
    plt.xlabel("Ziel-Schwierigkeitsklasse")
    plt.ylabel("Generator")
    plt.xticks(range(len(df.columns)), df.columns)
    plt.yticks(range(len(df)), df.index)
    plt.tight_layout()
    plt.savefig(os.path.join(TMP_PATH, 'duplicate_vis.png'))

    # Per Gen
    genframe = statframe.iloc[:-2]
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.xticks(range(len(genframe.index)), genframe.index)
    plt.bar(range(len(genframe.index)), genframe['sum_gen'], label='Summe Duplikate', color='orange', alpha=0.8)
    plt.bar(range(len(genframe.index)), genframe['gen_abort'], label='Anzahl Abbrüche', color='red', alpha=0.7)
    for step in range(int(min(genframe['sum_gen'])), int(max(genframe['sum_gen'])) + 1):
        plt.axhline(y=step, color='gray', linestyle='-', linewidth=0.8, zorder=-1, alpha=.5)
    plt.xlabel('Generator')
    plt.ylabel('Anzahl')
    plt.yticks(np.arange(0, round(genframe['sum_gen'].max())+1, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TMP_PATH, 'dup_per_gen.png'))

    # Per Goal
    goalframe = statframe.T.iloc[:-2]
    plt.figure(figsize=(10, 6))
    plt.xticks(range(len(goalframe.index)), goalframe.index)
    plt.bar(range(len(goalframe.index)), goalframe['sum_goal'], label='Summe Duplikate', color='orange', alpha=0.8)
    plt.bar(range(len(goalframe.index)), goalframe['goal_abort'], label='Anzahl Abbrüche', color='red', alpha=0.7)
    for step in range(int(min(goalframe['sum_goal'])), int(max(goalframe['sum_goal'])) + 1):
        plt.axhline(y=step, color='gray', linestyle='-', linewidth=0.8, zorder=-1, alpha=.5)
    plt.xlabel('Zielschwierigkeitsklasse')
    plt.ylabel('Anzahl')
    plt.yticks(np.arange(0, round(goalframe['sum_goal'].max())+1, 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(TMP_PATH, 'dup_per_goal.png'))

#getGenPlot()
#getDiffPlot()
#getDiffAdjustPlot()
#getDuplicateAbortPlot()

def getDiffFilePlot(path, barfont=False, fontsize=9, name="slice_per_diff", filtergoal=False):
    difficulty_classes = {f: os.path.join(path, f) for f in os.listdir(path)}
    difficulty_files = {f: len(os.listdir(p)) for f, p in difficulty_classes.items()}
    if filtergoal:
        newdict = {}
        for f, n in difficulty_files.items():
            for g in goal_list:
                if f in g:
                    newdict[f] = n
        difficulty_files = newdict
    df = pd.DataFrame(columns=["#slices"])
    for index, value in difficulty_files.items():
        df.loc[index] = value

    print(df)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df.index, df["#slices"], width=.9, label="Anzahl Abschnitte pro Schwierigkeitsklasse")
    plt.xlabel("Schwierigkeitsklasse")
    plt.ylabel("Anzahl Abschnitte")
    plt.xticks(df.index, fontsize=fontsize, rotation=90)  # Drehen der x-Achsentick-Beschriftungen für bessere Lesbarkeit
    if barfont:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')
    plt.legend()
    plt.tight_layout()  # Verbessert die Layout-Anordnung der Plots
    plt.savefig(os.path.join(TMP_PATH, name + '.png'))


getDiffFilePlot(DIFF_SLICE_PATH, True, filtergoal=True)
#getDiffFilePlot(ALT_DIFF_SLICE_PATH, True, None, "slice_per_diff_extratest")