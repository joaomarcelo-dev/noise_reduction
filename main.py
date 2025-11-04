import torch
import torchaudio
import noisereduce as nr
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings("ignore")

INPUT_FILE = "audio.wav"
OUTPUT_FILE = "voz_limpa_natural.wav"

# === 1. Separação Demucs ===
print("🔄 Carregando modelo Demucs...")
model = get_model('htdemucs_ft')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

audio, sr = torchaudio.load(INPUT_FILE)

if sr != 44100:
  print(f"   🔁 Convertendo {sr}Hz → 44100Hz")
  audio = torchaudio.transforms.Resample(sr, 44100)(audio)
  sr = 44100

if audio.shape[0] == 1:
  audio = audio.repeat(2, 1)

audio = audio / audio.abs().max()

print("🧠 Separando voz do ruído com Demucs...")
with torch.no_grad():
  sources = apply_model(model, audio.unsqueeze(0).to(device), device=device)[0]

vocals = sources[-1].cpu().mean(0).numpy()

# === 2. Filtro espectral leve ===
print("✨ Aplicando redução leve de ruído residual...")
clean_audio = nr.reduce_noise(
  y=vocals,
  sr=sr,
  prop_decrease=0.75,
  stationary=False,
  n_std_thresh_stationary=1.0
)

# === 3. Equalização suave (realce de presença e corte de graves pesados) ===
def butter_bandpass(lowcut, highcut, fs, order=2):
  nyq = 0.5 * fs
  low = lowcut / nyq
  high = highcut / nyq
  b, a = butter(order, [low, high], btype='band')
  return b, a

def apply_eq(data, fs):
  b, a = butter_bandpass(80, 12000, fs, order=2)  # remove subgraves e agudos extremos
  filtered = lfilter(b, a, data)
  # leve boost de presença 3–5 kHz
  presence = np.copy(filtered)
  presence_boost = np.exp(-0.5 * ((np.fft.rfftfreq(len(presence), 1/fs) - 4000)/1500)**2)
  spectrum = np.fft.rfft(presence)
  enhanced = np.fft.irfft(spectrum * (1 + 0.15 * presence_boost))
  return enhanced

clean_audio = apply_eq(clean_audio, sr)

# === 4. Normalização e limiter leve ===
max_amp = np.max(np.abs(clean_audio))

if max_amp > 0:
  clean_audio = clean_audio / max_amp * 0.98  # margem de 2% para evitar clipping

print("💾 Salvando voz limpa e natural...")
sf.write(OUTPUT_FILE, clean_audio, sr)
print(f"✅ Voz natural salva em: {OUTPUT_FILE}")
