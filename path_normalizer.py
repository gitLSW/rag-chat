from pathlib import Path

BANNED_CHARS = {'\\', ':', '*', '?', '"', '<', '>', '|', '\0'}


class PathNormalizer:

    def __init__(self, company_id):
        self.company_id = company_id
        
    def get_full_company_path(self, path):
        return merge_path(f"./{self.company_id}/docs", path)
    
    def get_relative_comany_path(self, path):
        root = Path(f'./{self.company_id}/docs').resolve()
        path = merge_path(root, path)
        return str(path.relative_to(root))



def merge_path(root, path):
    # Forbid these characters in paths (strictest security)
    if any(char in path for char in BANNED_CHARS):
        raise ValueError("Invalid path")
    
    root = Path(root).resolve()

    path = Path(path.lstrip("/"))
    file_name = path.stem.split('.')[0] + path.suffix  # remove extra extensions
    path = path.with_name(file_name)
    
    full_path = (root / path).resolve(strict=False)

    if not str(full_path).startswith(str(root)):
        raise ValueError("Path escapes root directory")
    
    return str(full_path)