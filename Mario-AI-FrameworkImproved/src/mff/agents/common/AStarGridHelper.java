package mff.agents.common;

import mff.agents.common.IGridHeuristic;
import mff.agents.common.IMarioAgentMFF;
import mff.agents.gridSearch.GridSearch;
import mff.agents.gridSearch.GridSearchNode;

import java.util.ArrayList;

public class AStarGridHelper {
    public static void giveLevelTilesWithPath(IMarioAgentMFF agent, String levelPath) {
        GridSearch gridSearch = GridSearch.initGridSearch(levelPath);
        ArrayList<GridSearchNode> gridPath = gridSearch.findGridPath();
        int[][] levelTilesWithPath = gridSearch.markGridPathInLevelTiles(gridPath);
        ((IGridHeuristic) agent).receiveLevelWithPath(levelTilesWithPath);
    }

    public static void giveGridPath(IMarioAgentMFF agent, String levelPath) {
        GridSearch gridSearch = GridSearch.initGridSearch(levelPath);
        ArrayList<GridSearchNode> gridPath = gridSearch.findGridPath();
        ((IGridWaypoints) agent).receiveGridPath(gridPath);
    }

    public static void showLevelTilesPath(String levelPath) {
        GridSearch gridSearch = GridSearch.initGridSearch(levelPath);
        ArrayList<GridSearchNode> gridPath = gridSearch.findGridPath();
        int[][] levelTilesWithPath = gridSearch.markGridPathInLevelTiles(gridPath);

        for (int height = 0; height < levelTilesWithPath[0].length; height++) {
            for (int[] column : levelTilesWithPath) {
                System.out.print(column[height]);
            }
            System.out.println();
        }
    }
}
