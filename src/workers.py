from typing import Any, Callable, Iterable
import subprocess, os, json
from queue import Queue
from threading import Thread

from .paths import PORTABLE_OLLAMA


status_queue: Queue = Queue()


def download_model(model_name: str) -> None:
    
    process = subprocess.Popen(
        [PORTABLE_OLLAMA, "pull", model_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    process.wait()

    status_queue.put({
        "download_model": model_name,
        "return": process.returncode,
    })
    
    
    
    
def run_worker(
    target: Callable[..., None], args: Iterable[Any] | None = None, 
    kwargs: dict[str, Any] | None = None, daemon: bool = True
    ) -> None:
    t: Thread = Thread(target=target, args=args, kwargs=kwargs, daemon=daemon)
    t.start()
    
    