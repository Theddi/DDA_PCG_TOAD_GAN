package mff.agents.common;

import mff.agents.gridSearch.GridSearchNode;

import java.util.ArrayList;

public interface IGridWaypoints {
    void receiveGridPath(ArrayList<GridSearchNode> gridPath);
}
