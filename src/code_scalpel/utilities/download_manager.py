# src/utils_tools/download_manager.py

import hashlib
import os
import time
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
from urllib3.util import Retry

# Retry configuration for requests
retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])


def download_file(
    url: str,
    save_path: str,
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    resume: bool = False,
    max_speed: Optional[float] = None,
    checksum: Optional[str] = None,
    auth: Optional[HTTPBasicAuth] = None,
    show_progress: bool = True,
) -> bool:
    """
    Downloads a file from a URL and saves it to the specified path.

    Args:
      url (str): The URL of the file to download.
      save_path (str): The path where the file should be saved.
      headers (Optional[Dict[str, str]]): Custom headers to include in the request.
      timeout (Optional[float]): The timeout value for the request (in seconds).
      resume (bool): Whether to resume an interrupted download or not.
      max_speed (Optional[float]): The maximum download speed in bytes per second.
      checksum (Optional[str]): The expected checksum of the file for integrity verification.
      auth (Optional[HTTPBasicAuth]): Authentication credentials for the request.
      show_progress (bool): Whether to show the progress bar or not.

    Returns:
      bool: True if the download was successful, False otherwise.
    """
    try:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Set the initial position for resuming downloads
        resume_position = (
            os.path.getsize(save_path) if resume and os.path.exists(save_path) else 0
        )

        # Configure the request session with retries and headers
        session = requests.Session()
        session.mount("http://", requests.adapters.HTTPAdapter(max_retries=retries))
        session.mount("https://", requests.adapters.HTTPAdapter(max_retries=retries))
        headers = headers or {}

        # Send the request and get the response
        response = session.get(
            url, headers=headers, timeout=timeout, stream=True, auth=auth
        )
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get("content-length", 0)) + resume_position
        block_size = 1024  # 1 KB

        with open(save_path, "ab") as f:
            if resume_position:
                f.seek(resume_position)
                f.truncate()

            if show_progress:
                pbar = tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading {save_path}",
                    initial=resume_position,
                    unit_divisor=1024,
                )
            else:
                pbar = None

            start_time = time.time()
            for data in response.iter_content(chunk_size=block_size):
                if max_speed:
                    time.sleep(len(data) / max_speed)
                f.write(data)
                if pbar:
                    pbar.update(len(data))
                    elapsed_time = time.time() - start_time
                    speed = pbar.n / elapsed_time if elapsed_time > 0 else 0
                    remaining_time = (total_size - pbar.n) / speed if speed > 0 else 0
                    pbar.set_postfix(
                        speed=f"{speed / 1024:.2f} KiB/s", eta=f"{remaining_time:.2f} s"
                    )

            if pbar:
                pbar.close()

        # Verify file integrity if checksum is provided
        if checksum and not verify_checksum(save_path, checksum):
            print(f"Checksum verification failed for {save_path}")
            return False

        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


def verify_checksum(
    file_path: str, expected_checksum: str, algorithm: str = "sha256"
) -> bool:
    """
    Verify the checksum of a file.

    Args:
      file_path (str): The path to the file.
      expected_checksum (str): The expected checksum value.
      algorithm (str): The hashing algorithm to use (default: 'sha256').

    Returns:
      bool: True if the checksum matches, False otherwise.
    """
    hash_func = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest() == expected_checksum
