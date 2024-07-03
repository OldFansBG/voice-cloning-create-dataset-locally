# RVC v2 Dataset Creation Tool
 
## Requirements

Make sure you have the following Python packages installed:
```bash
pip install yt-dlp ffmpeg-python numpy==1.26.4 librosa soundfile demucs
```

Additionally, make sure ffmpeg is installed on your system.

## Usage

To use the script, run it from the command line with the following syntax:

python process_audio.py <YouTube URL> <audio name>

    <YouTube URL>: The URL of the YouTube video you want to process.
    <audio name>: A name for the audio processing, which will be used for naming the files and folders.

## Example
```bash
python process_audio.py https://www.youtube.com/watch?v=DEqXNfs_HhY my_audio
```

## Output
The processed audio segments are stored in a directory dataset/<audio name>.
These segments are then packaged into a ZIP file named dataset_<audio_name>.zip located in the current directory.

## Credits

This script is based on the repository [voice-cloning-create-dataset](https://github.com/zsxkib/voice-cloning-create-dataset) by [zsxkib](https://github.com/zsxkib).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
