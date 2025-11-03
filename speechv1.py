import sounddevice as sd
import numpy as np
import pygame
import librosa
import queue
import threading

# ----------------------------
# SETTINGS
# ----------------------------
samplerate = 16000
blocksize = 2048
q = queue.Queue()

# ----------------------------
# PYGAME SETUP
# ----------------------------
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Real-Time Vowel Visualizer")
font = pygame.font.SysFont(None, 48)

images = {
    'a': pygame.image.load('mouths/a.png'),
    'e': pygame.image.load('mouths/e.png'),
    'i': pygame.image.load('mouths/i.png'),
    'o': pygame.image.load('mouths/o.png'),
    'u': pygame.image.load('mouths/u.png')
}

current_vowel = None
running = True

# ----------------------------
# AUDIO CALLBACK
# ----------------------------
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

stream = sd.InputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=audio_callback)
stream.start()

# ----------------------------
# FORMANT-BASED DETECTION
# ----------------------------
def estimate_formants(y, sr):
    y = librosa.effects.preemphasis(y)
    S = np.abs(librosa.stft(y, n_fft=512))
    freqs = librosa.fft_frequencies(sr=sr)
    spec = np.mean(S, axis=1)
    peaks = np.argpartition(spec, -3)[-3:]
    formants = sorted(freqs[peaks])[:3]
    return formants

vowel_formants = {
    'a': [(700, 900, 1100, 1500), (500, 800, 1200, 1800)],   # "a" in "cat", "father"
    'e': [(400, 600, 1800, 2500)],                           # "e" in "bed"
    'i': [(250, 400, 2000, 3200)],                           # "ee" in "see"
    'o': [(400, 700, 800, 1300), (500, 800, 900, 1500)],     # "o" in "hot", "go"
    'u': [(250, 400, 700, 1100), (300, 500, 900, 1300)]      # "oo" in "food", "put"
}

def classify_vowel(formants):
    if len(formants) < 2:
        return None
    f1, f2 = formants[:2]
    for vowel, ranges in vowel_formants.items():
        for (f1_min, f1_max, f2_min, f2_max) in ranges:
            if f1_min <= f1 <= f1_max and f2_min <= f2 <= f2_max:
                return vowel
    return None

def audio_processor():
    global current_vowel
    while running:
        audio = q.get().flatten()
        formants = estimate_formants(audio, samplerate)
        vowel = classify_vowel(formants)
        if vowel:
            current_vowel = vowel
            print(f"Detected vowel: {vowel} | formants: {formants}")

threading.Thread(target=audio_processor, daemon=True).start()

# ----------------------------
# MAIN LOOP
# ----------------------------
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))
    if current_vowel and current_vowel in images:
        screen.blit(images[current_vowel], (0, 0))
        img_text = font.render(f"Vowel: {current_vowel.upper()}", True, (255, 255, 255))
        screen.blit(img_text, (10, 10))
    pygame.display.flip()

stream.stop()
stream.close()
pygame.quit()