package mff.agents.gridSearch;

import java.text.CharacterIterator;
import java.text.StringCharacterIterator;
import java.util.ArrayList;

public class GridPathVisualizer {
    private static final int NUMBER_NOT_IN_LEVEL_TILE_VALUES = 42000;

    public static void visualizePath(String level, int[][] levelTiles, ArrayList<GridSearchNode> path) {
        int [][] levelTilesWithPath = levelTiles.clone();
        for (GridSearchNode node : path) {
            levelTilesWithPath[node.tileX][node.tileY] = NUMBER_NOT_IN_LEVEL_TILE_VALUES;
        }

        String l = level.stripTrailing();
        StringBuilder sb = new StringBuilder(l.length());
        CharacterIterator it = new StringCharacterIterator(l);
        int char_counter = 0;

        for (char c = it.first(); c != CharacterIterator.DONE; c = it.next()) {
            if (c == '\r' || c == '\n') {
                sb.append(c);
                continue;
            }

            if (levelTilesWithPath[char_counter % levelTiles.length][char_counter / levelTiles.length] == NUMBER_NOT_IN_LEVEL_TILE_VALUES)
                sb.append('■');
            else
                sb.append(c);

            char_counter++;
        }

        String result = sb.toString();
        System.out.println(result);
    }
}
