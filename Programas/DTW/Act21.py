# Reconocimiento de voz DTW

import numpy as np
import matplotlib.pyplot as plt
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from dtaidistance import dtw
import os
import tempfile
import warnings
from collections import defaultdict

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)

warnings.filterwarnings(
    "ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load"
)


class Reconocimiento:

    def __init__(self, sample_rate=22050, duration=3):

        self.sr = sample_rate
        self.duration = duration

    def record(self):

        print("Grabando...")
        recording = sd.rec(int(self.duration * self.sr), samplerate=self.sr, channels=1)
        sd.wait()
        print("Grabacion terminada")
        audio = recording.flatten()
        return audio

    def temp_audio(self, y):

        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        write(temp_file.name, self.sr, y)
        return temp_file.name

    def load_audio(self, file_path):

        try:
            y, _ = librosa.load(file_path, sr=self.sr)
        except:
            y, _ = librosa.audioread_load(file_path, sr=self.sr)

        y = librosa.effects.preemphasis(y)
        y = librosa.effects.trim(y, top_db=15)[0]
        y = librosa.util.normalize(y)
        return librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=15).flatten()

    def plot_results(self, ref_mfcc, test_mfcc, best_path, distance, paths):

        # Comparacion de audios
        plt.figure(figsize=(10, 5))
        plt.plot(ref_mfcc, label="Audio 1", alpha=0.7)
        plt.plot(test_mfcc, label="Audio 2", alpha=0.7)

        plt.text(
            0.5,
            0.9,
            f"Distancia DTW: {distance:.2f}",
            transform=plt.gca().transAxes,
            ha="center",
        )

        plt.title("Comparacion de audios con DTW")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Alineamiento de audios

        plt.figure(figsize=(10, 5))

        ax1 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
        ax1.imshow(paths, cmap="viridis", origin="lower", aspect="auto")
        ax1.plot(
            [p[1] for p in best_path], [p[0] for p in best_path], "r-", linewidth=2
        )
        ax1.set_xlabel("Audio 2")
        ax1.set_ylabel("Audio 1")
        ax1.set_title("Alineamiento con DTW")

        # Audio 1
        ax2 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        ax2.plot(ref_mfcc, "b", label="Audio 1")
        ax2.set_xticks([])
        ax2.legend()

        # Audio 2
        ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
        ax3.plot(test_mfcc, np.arange(len(test_mfcc)), "g", label="Audio 2")
        ax3.set_yticks([])
        ax3.legend()

        plt.tight_layout()
        plt.show()

    def similarity(self, y1, y2, plot=False):

        distance, paths = dtw.warping_paths(y1, y2)
        best_path = dtw.best_path(paths)

        if plot == True:

            self.plot_results(y1, y2, best_path, distance, paths)

        return distance, best_path

    def score(self, distance, best_path):

        return distance / len(best_path)

    def load_dataset(self, words_dir):

        references = defaultdict(list)

        for word in os.listdir(words_dir):
            word_dir = os.path.join(words_dir, word)
            if os.path.isdir(word_dir):
                for audio_file in os.listdir(word_dir):
                    if audio_file.endswith((".wav", ".mp3", ".m4a")):
                        file_path = os.path.join(word_dir, audio_file)
                        mfcc = self.load_audio(file_path)
                        references[word].append(mfcc)

        return references

    def recognize(self, test_audio, references, plot_best=False):

        test_mfcc = self.load_audio(test_audio)
        best_score = float("inf")
        best_word = None
        all_scores = {}
        best_match = None

        for word, ref_mfcc_list in references.items():
            min_word_score = float("inf")
            best_ref_mfcc = None

            for ref_mfcc in ref_mfcc_list:
                distance, best_path = self.similarity(ref_mfcc, test_mfcc)
                score = self.score(distance, best_path)

                if score < min_word_score:
                    min_word_score = score
                    best_ref_mfcc = ref_mfcc

            all_scores[word] = min_word_score

            if min_word_score < best_score:
                best_score = min_word_score
                best_word = word
                best_match = (best_ref_mfcc, test_mfcc, word)

        if plot_best == True and best_match is not None:

            ref_mfcc, test_mfcc, word = best_match
            self.similarity(ref_mfcc, test_mfcc, plot=True)

        return best_word, best_score, all_scores

    def run(self, references_dir, plot=True):

        references = self.load_dataset(references_dir)

        test_recording = self.record()
        test_file = self.temp_audio(test_recording)

        best_word, best_score, all_scores = self.recognize(
            test_file, references, plot_best=plot
        )

        print("Resultados: ")
        for word, score in all_scores.items():
            print(f"{word}: {score:.2f}")

        print(
            f"La palabra mas parecida es: {best_word} con un score de {best_score:.2f}"
        )

        os.remove(test_file)
        return best_word, best_score


DTW = Reconocimiento(duration=3)

referencias = "/Users/patrickbustamante/Library/CloudStorage/GoogleDrive-p317694@uach.mx/My Drive/GitRepos/RDP/Programas/DTW/Datos/referencias"

best_word, best_score = DTW.run(referencias, plot=True)
