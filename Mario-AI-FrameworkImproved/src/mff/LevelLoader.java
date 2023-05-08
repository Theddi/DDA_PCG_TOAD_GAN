package mff;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class LevelLoader {
    public static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
            return content;
        } catch (IOException ignored) {
            // try with working directory set one folder up
        }

        try {
            content = new String(Files.readAllBytes(Paths.get("Mario-AI-Framework" + filepath.substring(1))));
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
}
