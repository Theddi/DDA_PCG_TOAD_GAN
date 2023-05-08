
def make_env(width, height, color, path='environment.txt'):
    env = ""
    for h in range(height):
        for w in range(width):
            env += color
        env += "\n"
    with open(path, 'w') as file:
        file.truncate(0)
        file.write(env)


#make_env(80, 80 , 'N', "../contexts/flowers_env.txt") 
