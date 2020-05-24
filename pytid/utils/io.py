import glob
import os


def find_shared_objects(prefix: str) -> str:
    """
    Finds compiled C libraries available as shared objects under the build folder.
    :param prefix: The filename prefix for which you want to import
    :return: The relative path to the library.
    """
    found_paths = glob.glob(os.path.join("build", "**", f"{prefix}*"), recursive=True)
    if len(found_paths) == 0:
        raise ImportError("I could not find the shared object file. Have you run setup.py?")
    elif len(found_paths) > 1:
        raise ImportError(f"I found too many shared objects with prefix={prefix}. Found files: {found_paths}")
    return found_paths[0]
