package mff.agents.benchmark;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class ResultAnalyzer {
    public static void main(String[] args) throws IOException {
        printLostLevels("astarGrid");
    }

    public static void printLostLevels(String agentName) throws IOException {
        File benchmarkFolder = new File("Mario-AI-Framework/agent-benchmark");
        ArrayList<File> filesInBenchmarkFolder = new ArrayList<>(Arrays.asList(benchmarkFolder.listFiles()));
        ArrayList<File> filesWithAgentResults = filesInBenchmarkFolder.stream().filter(file -> file.getName().contains(agentName)).collect(Collectors.toCollection(ArrayList::new));
        for (File agentResults : filesWithAgentResults) {
            List<String> allLines = Files.readAllLines(agentResults.toPath());
            for (String line : allLines) {
                String[] tokens = line.split(",");
                if (tokens[1].equals("false"))
                    System.out.println(tokens[0]);
            }
        }
    }
}
