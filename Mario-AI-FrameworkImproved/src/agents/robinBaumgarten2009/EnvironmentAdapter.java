package agents.robinBaumgarten2009;

import engine.core.MarioForwardModel;
import engine.helper.SpriteType;

import java.util.HashMap;

public class EnvironmentAdapter {

	private MarioForwardModel model;
	public static final int HalfObsWidth = 11;
	public static final int HalfObsHeight = 11;

	public EnvironmentAdapter(MarioForwardModel model) {
		this.model = model;
	}

	public byte[][] getLevelSceneObservationZ(int zLevel) {
		assert(zLevel == 0);

		byte[][] ret = new byte[HalfObsWidth * 2][HalfObsHeight * 2];
		int MarioXInMap = (int) model.getMarioFloatPos()[0] / 16; //(int)mario.x/16;
		int MarioYInMap = (int) model.getMarioFloatPos()[1] / 16; //mario.y/16;

		int[][] levelTiles = model.getWorld().level.getLevelTiles();

		for (int y = MarioYInMap - HalfObsHeight, obsX = 0; y < MarioYInMap + HalfObsHeight; y++, obsX++)
		{
			for (int x = MarioXInMap - HalfObsWidth, obsY = 0; x < MarioXInMap + HalfObsWidth; x++, obsY++)
			{
				if (x >=0 /*  && x <= level.xExit */ && y >= 0 && y < model.getLevelFloatDimensions()[1] / 16 /*level.height*/)
				{
					ret[obsX][obsY] = ZLevelMapElementGeneralization((byte)levelTiles[x][y]/*level.map[x][y]*/, zLevel);
				}
				else
					ret[obsX][obsY] = 0;
//                if (x == MarioXInMap && y == MarioYInMap)
//                    ret[obsX][obsY] = mario.kind;
			}
		}
		return ret;
	}

	private byte ZLevelMapElementGeneralization(byte el, int ZLevel)
	{
		// TODO: convert all block values to old ones
		if (el == 0)
			return 0;
		assert(ZLevel == 0);
		switch (el) {
			case 6:  // brick, simple, without any surprise.
			case 7:  // brick with a hidden coin
			case 50:  // brick with a hidden flower/mushroom
			case 51:  // brick with hidden life up
				return 6; // prevents cheating
			case 11:       // question brick, contains coin
			case 8:       // question brick, contains flower/mushroom
			// does not really make sense, but follows original implementation
				return 11; // question brick, contains something
		}
		return el;
	}

	public float[] getEnemiesFloatPos() {
		float[] enemyPosAndType = model.getSpritesFloatPosAndType();
		// the original implementation takes the following sprites:
		// goombas, koopas, bullet bill, spikys, flower enemy, shell, mushroom
		// mushroom isnt an enemy, but we will stick to original implementation
		// filter sprites:
		int validSprites = 0;
		for (int i = 0; i < enemyPosAndType.length / 3; i++) {
			if (isValid(enemyPosAndType[3 * i]))
				validSprites++;
		}
		float[] enemyPositionsFiltered = new float[validSprites * 3];

		int spritesFound = 0;
		for (int i = 0; i < enemyPosAndType.length / 3; i++) {
			if (isValid(enemyPosAndType[3 * i])) {
				// convert sprite type
				enemyPositionsFiltered[spritesFound * 3] = spriteTypes.get(enemyPosAndType[3 * i]);
				enemyPositionsFiltered[spritesFound * 3 + 1] = enemyPosAndType[3 * i + 1];
				enemyPositionsFiltered[spritesFound * 3 + 2] = enemyPosAndType[3 * i + 2];
				spritesFound++;
			}
		}

		return enemyPositionsFiltered;

		/*ArrayList<Float> poses = new ArrayList<Float>();
		for (Sprite sprite : sprites)
		{
			// check if is an influenceable creature
			if (sprite.kind >= Sprite.KIND_GOOMBA && sprite.kind <= Sprite.KIND_MUSHROOM)
			{
				poses.add((float)sprite.kind);
				poses.add(sprite.x);
				poses.add(sprite.y);
			}
		}

		float[] ret = new float[poses.size()];

		int i = 0;
		for (Float F: poses)
			ret[i++] = F;

		return ret;*/
	}

	// 2009 types
	private static final float OLD_GOOMBA = 2;
	private static final float OLD_GOOMBA_WINGED = 3;
	private static final float OLD_RED_KOOPA = 4;
	private static final float OLD_RED_KOOPA_WINGED = 5;
	private static final float OLD_GREEN_KOOPA = 6;
	private static final float OLD_GREEN_KOOPA_WINGED = 7;
	private static final float OLD_BULLET_BILL = 8;
	private static final float OLD_SPIKY = 9;
	private static final float OLD_SPIKY_WINGED = 10;
	private static final float OLD_ENEMY_FLOWER = 12;
	private static final float OLD_SHELL = 13;
	private static final float OLD_MUSHROOM = 14;

	// current types
	private static final float GOOMBA = 2;
	private static final float GOOMBA_WINGED = 3;
	private static final float RED_KOOPA = 4;
	private static final float RED_KOOPA_WINGED = 5;
	private static final float GREEN_KOOPA = 6;
	private static final float GREEN_KOOPA_WINGED = 7;
	private static final float SPIKY = 8;
	private static final float SPIKY_WINGED = 9;
	private static final float BULLET_BILL = 10;
	private static final float ENEMY_FLOWER = 11;
	private static final float MUSHROOM = 12;
	private static final float SHELL = 14;

	private static HashMap<Float, Float> spriteTypes;

	static {
		spriteTypes = new HashMap<>();
		spriteTypes.put(GOOMBA, OLD_GOOMBA);
		spriteTypes.put(GOOMBA_WINGED, OLD_GOOMBA_WINGED);
		spriteTypes.put(RED_KOOPA, OLD_RED_KOOPA);
		spriteTypes.put(RED_KOOPA_WINGED, OLD_RED_KOOPA_WINGED);
		spriteTypes.put(GREEN_KOOPA, OLD_GREEN_KOOPA);
		spriteTypes.put(GREEN_KOOPA_WINGED, OLD_GREEN_KOOPA_WINGED);
		spriteTypes.put(SPIKY, OLD_SPIKY);
		spriteTypes.put(SPIKY_WINGED, OLD_SPIKY_WINGED);
		spriteTypes.put(BULLET_BILL, OLD_BULLET_BILL);
		spriteTypes.put(ENEMY_FLOWER, OLD_ENEMY_FLOWER);
		spriteTypes.put(MUSHROOM, OLD_MUSHROOM);
		spriteTypes.put(SHELL, OLD_SHELL);
	}

	private boolean isValid(float enemyType) {
		return enemyType == SpriteType.GOOMBA.getValue() ||
				enemyType == SpriteType.GOOMBA_WINGED.getValue() ||
				enemyType == SpriteType.RED_KOOPA.getValue() ||
				enemyType == SpriteType.RED_KOOPA_WINGED.getValue() ||
				enemyType == SpriteType.GREEN_KOOPA.getValue() ||
				enemyType == SpriteType.GREEN_KOOPA_WINGED.getValue() ||
				enemyType == SpriteType.BULLET_BILL.getValue() ||
				enemyType == SpriteType.SPIKY.getValue() ||
				enemyType == SpriteType.SPIKY_WINGED.getValue() ||
				enemyType == SpriteType.ENEMY_FLOWER.getValue() ||
				enemyType == SpriteType.SHELL.getValue() ||
				enemyType == SpriteType.MUSHROOM.getValue();
	}

	public float[] getMarioFloatPos() {
		return model.getMarioFloatPos();
	}
}
