import os
import sys
class Mapslicer:

    def __init__(self, levelpath, width=28):
        self.name = os.path.basename(levelpath)
        self.level_mat = []
        self.level = levelpath
        if ".txt" not in self.level:
            raise Exception("NoTextFile")
        self.width = width
        if self.width % 2 != 0:
            raise Exception("NoHalvableWidth")
        self.read_level()

    def read_level(self):
        with open(self.level, 'r') as file:
            for line in file:
                self.level_mat.append(line)
    '''
    slice_level slices the level input read by read_level into slices of size width
    Double the amount of slices will be created to better cover the whole map
    Therefore the level is iterated in width/2 steps with width size
    returns the path to the sliced files
    '''
    def slice_level(self):
        levelWidth = len(self.level_mat[0])
        levelHeight = len(self.level_mat)
        stepSize = int(self.width / 2)

        outputpath = "./slicedlevels/"+self.name+"_SLICED/"
        pathExist = os.path.exists(outputpath)
        if not pathExist:
            os.makedirs(outputpath)
        for i in range(0, levelWidth - self.width, stepSize):
            levelSlice = []
            for h in range(levelHeight):
                levelSlice.append("\n"+self.level_mat[h][i:i+self.width] if h != 0 else ""+self.level_mat[h][i:i+self.width])
            sliceFileName = self.name+"_slice"+str(int(i/stepSize))+".txt"
            with open(outputpath+sliceFileName, 'w') as file:
                file.writelines(levelSlice)
        return os.path.abspath(outputpath)

#slicer = Mapslicer(sys.argv[1])
#print("Sliced files found at: "+slicer.slice_level())
