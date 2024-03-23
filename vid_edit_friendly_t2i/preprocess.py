import webdataset as wds
from pathlib import Path
from tqdm import tqdm
import os
video_folder_path = Path("Tokenflow_adapter/vid_edit_friendly_t2i/MSRVTT/Video/")
output_path = Path("Tokenflow_adapter/vid_edit_friendly_t2i/MSRVTT/")

with wds.TarWriter(str(output_path / "dataset.tar")) as writer:
    for video_filename in tqdm(os.listdir(video_folder_path), desc="Packing videos"):
        if video_filename.endswith(".mp4"):
            video_path = video_folder_path / video_filename
            with open(video_path, "rb") as video_file:
                sample = {
                    "__key__": video_filename,
                    "mp4": video_file.read()
                }
                writer.write(sample)
