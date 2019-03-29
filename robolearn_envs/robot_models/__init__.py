import os


def get_urdf_path(urdf_name=None):
    if urdf_name is None:
        urdf_name = ''
    else:
        if not urdf_name.endswith('.urdf'):
            urdf_name += '.urdf'
    resdir = os.path.join(
        os.path.dirname(__file__),
        urdf_name
    )
    return resdir
