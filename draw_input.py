# Main file
# get input from text
# call generate input
# get path, save to txt or make generator
# control motors to draw image
from sketch_gen import generate_image
from get_pos import path_generator

def get_user_input():
    pass

def main():
    user_prompt = get_user_input()
    prompt = f"A detailed image of a {user_prompt} drawn with a single continuous line without picking up the pen"
    filename = "images/newest-image.png"
    generate_image(prompt, filename)
    path = []
    # Thread #1
    path_generator(filename, path)
    # Thread #2
    curr_pos = path[0]
    for pos in path:
        direction = pos - curr_pos
        # tell motors to go in one of 8 directions
        pass

if __name__ == "__main__":
    main()