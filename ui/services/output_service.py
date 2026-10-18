from pathlib import Path
from ..core.config import ROOT
import os

class OutputService:
    def inventory(self):
        """Efficiently count files and bytes in output directories using os.walk()."""
        roots = [ROOT / "datasets", ROOT / "models", ROOT / "output"]
        out = []
        for r in roots:
            if r.exists():
                file_count = 0
                total_size = 0
                try:
                    for dirpath, dirnames, filenames in os.walk(r):
                        file_count += len(filenames)
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            try:
                                total_size += os.path.getsize(filepath)
                            except OSError:
                                # File may have been deleted; skip it
                                pass
                except OSError:
                    # Permission denied or other I/O error
                    pass
                out.append((str(r.relative_to(ROOT)), file_count, total_size))
        return out

    def dataset_tree(self, did):
        """List files in a dataset using os.walk() for efficiency."""
        r = ROOT / "datasets" / f"dataset_{int(did):03d}"
        if not r.exists():
            return []
        
        files = []
        try:
            for dirpath, dirnames, filenames in os.walk(r):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    files.append(str(Path(filepath).relative_to(r)))
        except OSError:
            # Permission denied or other I/O error
            return []
        return files
