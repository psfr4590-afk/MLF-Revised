from ..core.config import ROOT
from command_center.security import validate_log_path, validate_tail_lines

class LogService:
 def files(self):
  roots=[ROOT/"datasets",ROOT/".runtime"]
  out=[]
  for r in roots:
   if r.exists():
    out += [p for p in r.rglob("*") if p.is_file() and "log" in p.name.lower() and not p.is_symlink()]
  return out
 def tail(self,path,lines=80):
  p=validate_log_path(path)
  n=validate_tail_lines(lines)
  return "".join(p.read_text(encoding="utf-8",errors="replace").splitlines(True)[-n:])
