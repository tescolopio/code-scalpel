# src/util_tools/process_manager.py

import subprocess
from typing import Tuple, Optional, Dict, Any

def run_command(command: str, 
                capture_output: bool = True, 
                input_data: Optional[str] = None,
                env: Optional[Dict[str, str]] = None,
                cwd: Optional[str] = None,
                timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Runs an external command and returns the output.

    Args:
      command (str): The command to run (as a string).
      capture_output (bool): Whether to capture the output (default: True).
      input_data (Optional[str]): Input data to provide to the process (default: None).
      env (Optional[Dict[str, str]]): Environment variables to set for the process (default: None).
      cwd (Optional[str]): Working directory to run the command in (default: None).
      timeout (Optional[float]): Timeout for the command (default: None).

    Returns:
      Tuple[int, str, str]: A tuple containing:
        - The return code of the process.
        - The standard output (stdout) as a string.
        - The standard error (stderr) as a string.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,  # Use shell to interpret the command
            capture_output=capture_output,
            text=True,  # Capture output as text
            input=input_data,
            env=env,
            cwd=cwd,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return 1, "", f"Error: Command not found: {command}"
    except subprocess.TimeoutExpired:
        return 1, "", f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", f"Error running command: {e}"

def terminate_process(process: subprocess.Popen) -> bool:
    """
    Terminates a running process.

    Args:
      process (subprocess.Popen): The process to terminate.

    Returns:
      bool: True if the process was terminated successfully, False otherwise.
    """
    try:
        process.terminate()
        process.wait(timeout=5)
        return True
    except Exception as e:
        print(f"Error terminating process: {e}")
        return False

def get_process_status(process: subprocess.Popen) -> str:
    """
    Gets the status of a process.

    Args:
      process (subprocess.Popen): The process to check.

    Returns:
      str: The status of the process ('running', 'terminated', or 'unknown').
    """
    if process.poll() is None:
        return 'running'
    elif process.returncode is not None:
        return 'terminated'
    else:
        return 'unknown'

def wait_for_process(process: subprocess.Popen, timeout: Optional[float] = None) -> Tuple[int, str, str]:
    """
    Waits for a process to finish and returns the output.

    Args:
      process (subprocess.Popen): The process to wait for.
      timeout (Optional[float]): Timeout for waiting (default: None).

    Returns:
      Tuple[int, str, str]: A tuple containing:
        - The return code of the process.
        - The standard output (stdout) as a string.
        - The standard error (stderr) as a string.
    """
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return process.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        process.terminate()
        stdout, stderr = process.communicate()
        return 1, stdout, f"Error: Process timed out after {timeout} seconds"

def sanitize_command(command: str) -> str:
    """
    Sanitizes a command to prevent command injection vulnerabilities.

    Args:
      command (str): The command to sanitize.

    Returns:
      str: The sanitized command.
    """
    # Simple sanitization by escaping special characters
    return command.replace(";", r"\;").replace("&", r"\&").replace("|", r"\|")