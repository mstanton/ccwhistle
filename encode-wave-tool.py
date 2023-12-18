import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pygame
from pydub import AudioSegment
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

# Initialize pygame.mixer
pygame.mixer.init()

# Initialize variables for the audio file
y = None
sr = None

def export_riff_protocol_pydub(wavefile_path):
    # Load the WAV file using pydub
    audio = AudioSegment.from_wav(wavefile_path)

    # Get RIFF information
    riff_info = {
        'Channels': audio.channels,
        'SampleWidth': audio.sample_width,
        'FrameRate': audio.frame_rate,
        'NumFrames': len(audio),
        'CompressionType': audio.frame_width,
        'CompressionName': 'not available',  # pydub does not provide compression information
    }

    # Export RIFF information to a text file with progress indicator
    text_file_path = wavefile_path.replace('.wav', '_riff_info.txt')
    with open(text_file_path, 'w') as text_file, tqdm(total=len(riff_info), desc="Exporting RIFF Information") as pbar:
        for key, value in riff_info.items():
            text_file.write(f'{key}: {value}\n')
            pbar.update(1)

    print(f'RIFF information exported to: {text_file_path}')
    
    return text_file_path

def display_riff_file(riff_text_file):
    try:
        # Read and display the contents of the RIFF file
        with open(riff_text_file, 'r') as file:
            riff_content = file.read()
            print('\nRIFF Information:\n')
            print(riff_content)
    except Exception as e:
        print(f"Error displaying RIFF information: {e}")

def load_audio_file(wavefile_path):
    global y, sr
    try:
        print("Loading audio file...")
        # Load the audio file
        y, sr = librosa.load(wavefile_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")

def plot_mel_spectrogram(wavefile_path):
    global y, sr
    try:
        # Export RIFF protocol to a text file using pydub
        riff_text_file = export_riff_protocol_pydub(wavefile_path)
    except Exception as e:
        print(f"Error exporting RIFF information: {e}")
        return

    try:
        display_riff_file(riff_text_file)
    except Exception as e:
        print(f"Error displaying RIFF information: {e}")

    # Load the audio file
    load_audio_file(wavefile_path)

    if y is None or sr is None:
        print("Audio file not loaded. Load audio file first.")
        return

    try:
        print("Computing mel spectrogram...")
        # Compute the mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024, n_mels=128)

        # Convert to decibels
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 5))
        im = librosa.display.specshow(mel_spectrogram_db[:, :10], x_axis='time', y_axis='mel', sr=sr, hop_length=1024, cmap='viridis')

        # Add sliders and button for audio controls
        ax_time = plt.axes([0.1, 0.01, 0.65, 0.03])
        s_time = Slider(ax_time, 'Time', 0, mel_spectrogram_db.shape[1] - 1, valinit=0, valstep=1)

        def update(frame):
            im.set_array(mel_spectrogram_db[:, frame * 10: (frame + 1) * 10])
            return [im]

        ani = FuncAnimation(fig, update, frames=mel_spectrogram_db.shape[1] // 10, interval=50, blit=True, repeat=False)

        def update_audio(val):
            frame = int(s_time.val)

            # Check if music is currently playing
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.set_pos(frame * 10 / sr)
            else:
                print("Music isn't playing. Start playing music first.")
                return  # Return without attempting to play if music isn't loaded

            # Check if music is loaded
            if not pygame.mixer.music.get_pos():
                print("Music not loaded. Load music first.")
                return

            pygame.mixer.music.play()

        def pause_audio(val):
            pygame.mixer.music.pause()

        def resume_audio(val):
            pygame.mixer.music.unpause()

        # Load the audio file and play it
        pygame.mixer.music.load(wavefile_path)
        pygame.mixer.music.play()

        s_time.on_changed(update_audio)

        ax_pause = plt.axes([0.8, 0.01, 0.1, 0.03])
        ax_resume = plt.axes([0.92, 0.01, 0.1, 0.03])

        b_pause = Button(ax_pause, 'Pause')
        b_pause.on_clicked(pause_audio)

        b_resume = Button(ax_resume, 'Resume')
        b_resume.on_clicked(resume_audio)

        # Explicitly pass 'ax' argument to colorbar to avoid warning
        plt.colorbar(mappable=im, ax=ax, format='%+2.0f dB')
        plt.title('Mel Spectrogram Animation')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        plt.show()

    except Exception as e:
        print(f"Error computing mel spectrogram: {e}")
        return

# Replace 'your_audio_file.wav' with the path to your WAV file
plot_mel_spectrogram('2023-12-16_21h51m41s.wav')
