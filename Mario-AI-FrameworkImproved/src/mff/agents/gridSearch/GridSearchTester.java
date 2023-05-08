package mff.agents.gridSearch;

import java.util.ArrayList;

public class GridSearchTester {
    public static void main(String[] args) {
        testOriginalLevels();
        testGeneratedLevels();
    }

    private static void testOriginalLevels() {
        for (int i = 1; i <= 15; i++) {
            System.out.println("Testing grid search for: original-" + i);
            GridSearchMain.findGridPathForLevel("./levels/original/lvl-" + i + ".txt", 0, false);
        }
    }

    private static void testGeneratedLevels() {
        for (String levelGenerator : levelGenerators) {
            for (int i = 1; i <= 100; i++) {
                System.out.println("Testing grid search for: " + levelGenerator + "-" + i);
                GridSearchMain.findGridPathForLevel("./levels/" + levelGenerator + "/lvl-" + i + ".txt", 0, false);
            }
        }
    }

    private static final ArrayList<String> levelGenerators = new ArrayList<>() {{
        add("krys");
        add("ge");
        add("hopper");
        add("notch");
        add("notchParam");
        add("notchParamRand");
        add("ore");
        add("patternCount");
        add("patternOccur");
        add("patternWeightCount");
    }};
}
