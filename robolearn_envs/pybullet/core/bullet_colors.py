pb_colors = {
    'orange': (1., 0.5088, 0.0468, 1),
    'red': (1., 0., 0., 1),
    'blue': (0., 0., 1., 1),
    'green': (0., 1., 0., 1),
    'black': (0., 0., 0., 1),
    'white': (1., 1., 1., 1),
    'gray': (.5, .5, .5, 1),
    'yellow': (1., 1., 0., 1),
    'purple': (1., 0., 1., 1),
    'turquoise': (0., 1., 1., 1),
    'indigo': (0.33, 0., 0.5, 1),
    # 'dark_gray': (.175, .175, .175, 1),
    'dark_gray': (.3, .3, .3, 1),
    'flat_black': (0.1, 0.1, 0.1, 1),
    'red_bright': (.87, 0.26, 0.07, 1),
    'sky_blue': (0.13, 0.44, 0.70, 1),
    'zinc_yellow': (0.9725, 0.9529, 0.2078, 1),
    'dark_yellow': (0.7, 0.7, 0., 1),
    'pink': (0.9, 0.4, 0.6, 1),
}


def get_pb_color(color):
    if color.lower() not in pb_colors:
        raise ValueError("Color %s is not available. Choose one of "
                         "the fowllowing: %s"
                         % (color, [cc for cc in pb_colors.keys()]))
    return pb_colors[color]
