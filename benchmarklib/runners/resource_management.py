import multiprocessing
import psutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from enum import Enum

class FailureReason(Enum):
    MEMORY_LIMIT = "memory_limit_exceeded"
    TIMEOUT = "timeout_exceeded"
    ERROR = "execution_error"

@dataclass
class RunResult:
    success: bool
    result: Any = None
    failure_reason: Optional[FailureReason] = None
    memory_used_mb: Optional[float] = None
    error_message: Optional[str] = None

def _worker_wrapper(func, args, kwargs, result_queue):
    """Wrapper to catch exceptions and return them via queue"""
    try:
        result = func(*args, **kwargs)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))

def run_with_resource_limits(
    func: Callable,
    args: tuple = tuple(),
    kwargs: dict = dict(),
    memory_limit_mb: Optional[float] = None,
    timeout_seconds: Optional[float] = None,
    check_interval: float = 10.0
):
    """
    Run a function in a separate process with memory and time limits.
    
    Args:
        func: Function to execute
        memory_limit_mb: Maximum memory in MB (None for no limit)
        timeout_seconds: Maximum execution time (None for no limit)
        check_interval: Interval to check memory and timeout (default: 10.0)
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        RunResult with success status and result or failure info
    """
    result_queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_worker_wrapper,
        args=(func, args, kwargs, result_queue)
    )
    
    process.start()
    start_time = time.time()
    max_memory = 0
    
    try:
        proc = psutil.Process(process.pid)
    except psutil.NoSuchProcess:
        proc = None

    # Monitor the process
    last_check = time.time()
    while proc is not None and process.is_alive() and result_queue.empty():
        try:
            if time.time() - last_check < check_interval:
                # don't poll memory too often
                # but do check for results & process aliveness twice per second
                time.sleep(0.5)
                continue
            
            last_check = time.time()

            # Check memory
            if memory_limit_mb is not None:
                mem_info = proc.memory_info()
                memory_mb = mem_info.rss / (1024 * 1024)
                max_memory = max(max_memory, memory_mb)
                
                if memory_mb > memory_limit_mb:
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                    return RunResult(
                        success=False,
                        failure_reason=FailureReason.MEMORY_LIMIT,
                        memory_used_mb=memory_mb,
                        error_message=f"Exceeded memory limit: {memory_mb:.2f} MB > {memory_limit_mb} MB"
                    )
            
            # Check timeout
            if timeout_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    process.terminate()
                    process.join(timeout=2)
                    if process.is_alive():
                        process.kill()
                    return RunResult(
                        success=False,
                        failure_reason=FailureReason.TIMEOUT,
                        memory_used_mb=max_memory if max_memory > 0 else None,
                        error_message=f"Exceeded timeout: {elapsed:.2f}s > {timeout_seconds}s"
                    )
                        
        except psutil.NoSuchProcess:
            break
    
    # Process finished, get result
    process.join(timeout=1)
    
    if result_queue.empty():
        return RunResult(
            success=False,
            failure_reason=FailureReason.ERROR,
            error_message="No result returned from process"
        )
    
    status, data = result_queue.get()
    
    if status == "success":
        return RunResult(
            success=True,
            result=data,
            memory_used_mb=max_memory if max_memory > 0 else None
        )
    else:
        return RunResult(
            success=False,
            failure_reason=FailureReason.ERROR,
            error_message=data,
            memory_used_mb=max_memory if max_memory > 0 else None
        )