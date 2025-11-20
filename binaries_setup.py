from typing import Any
import platform, requests, os, argparse, subprocess
import shutil, tarfile, zipfile, tempfile

import tomli
from tqdm import tqdm
from pathlib import Path


ARCHIVE_TYPES: tuple[str, ...] = ()
ROOT: Path = Path(__file__).parent


def prepend_path(path_to_add: str) -> dict:
    """Return a copy of os.environ with path_to_add placed at the front of PATH."""
    env = os.environ.copy()
    env["PATH"] = path_to_add + os.pathsep + env["PATH"]
    return env

def safe_extract_archive(archive_path: str, dest_dir: str) -> None:
    archive_path: Path = Path(archive_path)
    dest_dir: Path = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    temp_dir: Path = dest_dir / "__extract_tmp__"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    lower_name: str = archive_path.name.lower()

    if lower_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            members = z.infolist()
            for member in tqdm(members, desc="Extracting ZIP"):
                z.extract(member, temp_dir)

    elif lower_name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2")):
        with tarfile.open(archive_path, "r:*") as t:
            members = t.getmembers()
            for member in tqdm(members, desc="Extracting TAR"):
                t.extract(member, temp_dir)

    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    top_items: list[Path] = list(temp_dir.iterdir())

    if len(top_items) == 1 and top_items[0].is_dir():
        # strip "wrapper folder"
        wrapper: Path = top_items[0]
        for item in wrapper.iterdir():
            shutil.move(str(item), dest_dir)
    else:
        # Keep structure unchanged
        for item in top_items:
            shutil.move(str(item), dest_dir)

    shutil.rmtree(temp_dir)


def download_and_install(url: str, install_dir: str, backup_old: bool = True) -> None:
    suffix: str = Path(url).suffix 

    if url.lower().endswith((".tar.gz", ".tgz")):
        suffix = ".tar.gz"
    
    install_dir: Path = Path(install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    old_install : Path | None = None
    if backup_old and install_dir.exists() and any(install_dir.iterdir()):
        old_install = install_dir.with_suffix(".backup")
        if old_install.exists():
            shutil.rmtree(old_install)
        shutil.move(str(install_dir), str(old_install))
        install_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_path = Path(tmp_file.name)
        print(f"Downloading {url} to temp file {tmp_path}")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with tqdm(total=total, unit="B", unit_scale=True, desc="Pulling Assets") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                    pbar.update(len(chunk))

    try:
        safe_extract_archive(str(tmp_path), str(install_dir))
    except Exception as e:
        print(f"Extraction failed: {e}")

        # revert if extraction fails
        if backup_old and old_install:
            print("Restoring previous version...")
            shutil.rmtree(install_dir)
            shutil.move(str(old_install), str(install_dir))
        raise

    tmp_path.unlink(missing_ok=True)

    if backup_old and old_install:
        shutil.rmtree(old_install)

    print("Install complete.")


def load_releases_url(owner_name: str, project_name: str, version: str | None = None) -> str:
    global ARCHIVE_TYPES
    
    url: str = f"https://api.github.com/repos/{owner_name}/{project_name}/releases/latest"
    if version is not None:
        url = f"https://api.github.com/repos/{owner_name}/{project_name}/releases//tags/{version}"
        
    response: requests.Response = requests.get(url)
    data: dict[str, Any] = response.json()
    
    return [
        asset["browser_download_url"]
        for asset in data.get("assets", [])
        if asset["name"].endswith(ARCHIVE_TYPES)
    ]
    
    
def load_config() -> dict[str, Any]:
    if not os.path.exists(ROOT / "binaries_setup.toml"):
        raise FileNotFoundError()
    
    with open(ROOT / "binaries_setup.toml", "rb") as file:
        return tomli.load(file)
    

def make_script(path: str, content: str) -> None:
    with open(path, "w") as file:
        file.write(content)


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True)


def install_model(script_path: str, model_url: str, name: str, is_unix: bool = False) -> None:
    if not is_unix:
        run_cmd([script_path, "pull", model_url])
        run_cmd([script_path, "cp", model_url, name])
        run_cmd([script_path, "rm", model_url])
        return
        
    run_cmd(["bash", script_path, "pull", model_url])
    run_cmd(["bash", script_path, "cp", model_url, name])
    run_cmd(["bash", script_path, "rm", model_url])
    
    
def get_first_match(key: str, data: list[str])-> str:
    return next(filter(lambda x: key in x, data), None)


def main() -> None:
    
    key: str
    pandoc_url: str
    ollama_url: str
    
    global ARCHIVE_TYPES
    
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download and install a binary archive with automatic extraction."
    )

    parser.add_argument(
        "--os", required=True,
        choices=["linux", "darwin", "windows"]
    )

    parser.add_argument("--pandoc-version", help="version of pandoc to install")
    parser.add_argument("--ollama-version", help="version of ollama to install")

    args: argparse.Namespace = parser.parse_args()
    
    operating_system: str = args.os
    configs: dict[str, Any] = load_config()
    pandoc_owner: str = configs["Url-Vars"]["pandoc-owner"]
    ollama_owner: str = configs["Url-Vars"]["ollama-owner"]
    scripts: dict[str, Any] = configs["Scripts"]
    ARCHIVE_TYPES = tuple(configs["Url-Vars"]["archive-types"])
    
    arch: str = platform.machine().lower()
    
    pandoc_urls: list[str] = load_releases_url(
        pandoc_owner, "pandoc", 
        args.pandoc_version if args.pandoc_version else None
    )
    
    ollama_urls: list[str] = load_releases_url(
        ollama_owner, "ollama", 
        args.ollama_version if args.ollama_version else None
    )
    
    binary_path: Path = ROOT / "data" / "bin"
    ollama_portable: Path = binary_path / "ollama" / "ollama_portable"
    
    if operating_system == "windows":
        prepend_path(binary_path / "ollama" )
    elif operating_system == "darwin":
        prepend_path(binary_path / "ollama" )
    elif operating_system == "linux":
        prepend_path(binary_path / "ollama" / "bin" )
    

    if not os.path.exists(binary_path / "ollama"):
        os.makedirs(binary_path / "ollama")
    else:
        try:
            run_cmd([str(ollama_portable), "stop"])
        except Exception as e:
            pass
    if not os.path.exists(binary_path / "pandoc"):
        os.makedirs(binary_path / "pandoc")

    if operating_system == "windows":
        if arch in ("amd64", "arm64"):
            key = f"{operating_system}-x86_64"
            pandoc_url = get_first_match(key, pandoc_urls)
            key = f"{operating_system}-{arch}."
            ollama_url = get_first_match(key, ollama_urls)
    elif operating_system == "darwin":
        key = f"{operating_system}."
        ollama_url = get_first_match(key, ollama_urls)
        if arch in ("amd64", "arm64"):
            key = f"{'x86_64'if arch == 'amd64' else arch}-macOS"
            pandoc_url = get_first_match(key, pandoc_urls)
    elif operating_system == "linux":
        key = f"{operating_system}-{arch}."
        pandoc_url = get_first_match(key, pandoc_urls)
        ollama_url = get_first_match(key, ollama_urls)
        #if arch in "arm64":
        
    download_and_install(pandoc_url, binary_path / "pandoc")  
    download_and_install(ollama_url, binary_path / "ollama") 
    
    ext: str
        
    if operating_system == "windows":
        ext = ".bat"
        make_script(ollama_portable.with_suffix(ext), scripts["ollama-windows"])
    elif operating_system == "darwin":
        ext = ".sh"
        make_script(ollama_portable.with_suffix(ext), scripts["ollama-macOS"])
    elif operating_system == "linux":
        ext = ".sh"
        make_script(ollama_portable.with_suffix(ext), scripts["ollama-unix"])
        #if arch in "arm64":

    current_os: str = platform.system().lower()
    if operating_system == current_os:
        model_sec: dict[str, str] = configs["Models"]
        is_unix_like: bool = current_os in ("darwin", "linux")
        script_path: str = str(ollama_portable.with_suffix(ext))
        if is_unix_like:            
            run_cmd(["chmod", "+x", script_path])
            
        print("\nDownloading Embedding Model:")
        install_model(script_path, model_sec["embedding-pull"], model_sec["embedding-name"], is_unix_like)
        print("Download Finished\n")
        
        print("Downloading LLM:")
        install_model(script_path, model_sec["llm-pull"], model_sec["llm-name"], is_unix_like)
        print("Download Finished\n")
        

    
if __name__ == "__main__":
    main()