from typing import Any
import platform, requests, os, argparse, subprocess
import shutil, tarfile, zipfile, tempfile

import tomli
from tqdm import tqdm
from pathlib import Path


ROOT: Path = Path(__file__).parent


def run_cmd(args: list[str], get_output: bool = False) -> str | None:
    if get_output:
        result = subprocess.run(
            args, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True
        )
        return result.stdout
        
    subprocess.run(args, check=True)
    



def install_model(script_path: str, model_url: str, name: str | None = None, is_unix: bool = False) -> None:
    if not is_unix:
        run_cmd([script_path, "pull", model_url])
        if name is None:
            return
        run_cmd([script_path, "cp", model_url, name])
        run_cmd([script_path, "rm", model_url])
        return
        
    run_cmd(["bash", script_path, "pull", model_url])
    if name is None:
        return
    run_cmd(["bash", script_path, "cp", model_url, name])
    run_cmd(["bash", script_path, "rm", model_url])
    
    
    
def main() -> None:
    
    ext: str
    binary_path: Path = ROOT / "data" / "bin"
    ollama_portable: Path | str = binary_path / "ollama" / "ollama_portable"
    current_platform: str = platform.system().lower()
    
    if current_platform == "windows":
        ext = ".bat"   
    elif current_platform in ("darwin", "linux"):
        ext = ".sh"
        
    ollama_portable = str(ollama_portable.with_suffix(ext))  
    if not os.path.exists(ollama_portable):
        print("binaries have not been set up")
    
      
    print(run_cmd([ollama_portable, "list"], get_output=True))
    arg_in: str = input("> ")
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="pull or delete ollama models"
    )

    parser.add_argument("action", choices=["pull", "del"])
    parser.add_argument("pull_or_name", help="pull url of model or name of model wanting to delete")
    parser.add_argument("name", nargs="?", default=None, help="name to rename model")

    args: argparse.Namespace = parser.parse_args(arg_in.split())
    
    if args.action == "del":
        run_cmd([ollama_portable, "rm", args.pull_or_name])
    elif args.action == "pull": 
        install_model(ollama_portable, args.pull_or_name, args.name, (current_platform in ("darwin", "linux")))
    
    
    
    
if __name__ == "__main__":
    main()
    
        
    
    
    
    