package agents.robinBaumgarten2009;

import engine.core.MarioGame;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class robinBaumgarten2009Test {
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
        }
        catch (IOException e) {
            System.out.println("Level couldn't be loaded, please check the path provided with regards to your working directory.");
            System.exit(1);
        }
        return content;
    }

    // TODO: not finished, missing at least block values conversion and flower enemy adaptation

    public static void main(String[] args) {
        MarioGame game = new MarioGame();
        game.runGame(new agents.robinBaumgarten2009.AStarAgent(), getLevel("./levels/original/lvl-1.txt"),
                200, 0 , true);
    }
}
