'''
Helper file to save sequence of images as a GIF
'''

def save_gif(frames, path, duration=40, loop=0):
    """
    Save a list of image frames as a GIF.
    """
    if not frames:
        return False

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration, loop=loop)

    return True
