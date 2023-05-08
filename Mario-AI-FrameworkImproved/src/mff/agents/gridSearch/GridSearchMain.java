package mff.agents.gridSearch;

import engine.core.MarioEvent;
import engine.core.MarioWorld;
import mff.LevelLoader;

import java.util.ArrayList;

public class GridSearchMain {
    public static void main(String[] args) {
        findGridPathForLevel("./levels/original/lvl-1.txt", 0, true);
    }

    public static void findGridPathForLevel(String levelName, int horizontalJumpBoost, boolean verbose) {
        String level = LevelLoader.getLevel(levelName);
        MarioEvent[] killEvents = new MarioEvent[0];
        MarioWorld world = new MarioWorld(killEvents);
        world.initializeLevel(level, 1000000);
        int[][] levelTiles = world.level.getLevelTiles();
        int marioTileX = world.level.marioTileX;
        int marioTileY = world.level.marioTileY;

        GridSearch gridSearch = new GridSearch(levelTiles, marioTileX, marioTileY, horizontalJumpBoost);
        ArrayList<GridSearchNode> resultPath = gridSearch.findGridPath();

        if (verbose) {
            System.out.println("Total nodes visited: " + gridSearch.totalNodesVisited);
            GridPathVisualizer.visualizePath(level, levelTiles, resultPath);
        }

        if (!gridSearch.success && !verbose) {
            System.out.println("Grid search failed:");
            GridPathVisualizer.visualizePath(level, levelTiles, resultPath);
        }
    }
}
