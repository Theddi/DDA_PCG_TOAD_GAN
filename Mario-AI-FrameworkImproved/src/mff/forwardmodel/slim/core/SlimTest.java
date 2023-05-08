package mff.forwardmodel.slim.core;

import engine.core.MarioLevelGenerator;
import engine.core.MarioLevelModel;
import engine.core.MarioTimer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

public class SlimTest {
    private static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
            return content;
        } catch (IOException ignored) {
            // try with working directory set one folder down
        }
        try {
            content = new String(Files.readAllBytes(Paths.get("." + filepath)));
        } catch (IOException e) {
            System.out.println("Level couldn't be loaded, please check the path provided with regards to your working directory.");
            System.exit(1);
        }
        return content;
    }

    public static void main(String[] args) {
        //correctnessTest();
        advanceSpeedTest();
    }

    private static void correctnessTest() {
        for (int i = 1; i < 16; i++) {
            MarioGameSlim game = new MarioGameSlim(true, false);
            game.runGame(new mff.agents.astar.Agent(), getLevel("./levels/original/lvl-" + i + ".txt"), 200, 0, false);
        }
    }

    private static final ArrayList<String> levelTypes = new ArrayList<>() {{
        add("original");
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

    private static void advanceSpeedTest() {
        for (String levelType : levelTypes) {
            int levelCount;
            if (levelType.equals("original"))
                levelCount = 15;
            else
                levelCount = 100;

            for (int i = 1; i <= levelCount; i++) {
                double originalTime = 0;
                double slimTime = 0;
                double slimWindowTime = 0;

                String level = getLevel("./levels/" + levelType + "/lvl-" + i + ".txt");

//                for (int j = 0; j < 1; j++) {
//                    MarioGameSlim game = new MarioGameSlim(false, true);
//                    game.runGame(new mff.agents.astar.Agent(), level, 30, 0, false);
//                }
                for (int k = 0; k < 1; k++) {
                    MarioGameSlim game = new MarioGameSlim(false, true);
                    TestResult testResult = game.runGame(new mff.agents.astar.Agent(), level, 30, 0, false);
                    originalTime += testResult.originalTime;
                    slimTime += testResult.slimTime;
                    slimWindowTime += testResult.slimWindowTime;
                }

                System.out.println(levelType + "-" + i + "," + originalTime + "," + slimTime + "," + slimWindowTime);
            }
        }
    }
}
