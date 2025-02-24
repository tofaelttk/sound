import cv2
import mediapipe as mp
import numpy as np
import math
import time
import threading
import os
import sounddevice as sd

# Global audio parameters (controlled by hand gestures)
sample_rate = 44100
current_freq = 440.0    # Frequency from right-hand gesture (200-2000 Hz)
current_volume = 0.0    # Amplitude (0.0 to 1.0) from distance between hands
current_beats = 120.0   # BPM from left-hand gesture (60-180 BPM)

# --- Helper: 8D Stereo Effect ---
def apply_8d_effect(mono_signal, sr, pan_speed=0.25):
    """
    Apply a continuously modulated stereo panning effect (8D-like) to a mono signal.
    The panning oscillates over time, giving the impression that the sound swirls around.
    """
    t = np.arange(len(mono_signal)) / sr
    pan = np.sin(2 * np.pi * pan_speed * t)  # oscillates between -1 and 1
    left = mono_signal * np.sqrt((1 - pan) / 2)
    right = mono_signal * np.sqrt((1 + pan) / 2)
    stereo = np.column_stack((left, right))
    return stereo

# --- Audio Synthesis Functions ---
def generate_kick(freq, duration, sr):
    """Generate a kick drum: a sine wave that quickly decays and drops in pitch."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    f0 = freq
    f1 = freq * 0.5
    k = np.log(f1 / f0) / duration
    instantaneous_freq = f0 * np.exp(k * t)
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / sr
    envelope = np.exp(-20 * t)
    kick = envelope * np.sin(phase)
    return kick

def generate_snare(duration, sr):
    """Generate a snare drum using white noise with a fast-decay envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    envelope = np.exp(-30 * t)
    snare = envelope * noise
    return snare

def generate_synth(freq, duration, sr, volume):
    """Generate a percussive synth note with a square wave (boxy sound)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    note = np.sign(np.sin(2 * np.pi * freq * t))
    envelope = np.exp(-15 * t)
    synth = volume * envelope * note
    return synth

def generate_hihat(duration, sr):
    """Generate a hi-hat using high-frequency white noise with a very fast decay."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    envelope = np.exp(-50 * t)
    hihat = envelope * noise
    return hihat

def generate_clap(duration, sr):
    """Generate a clap sound using white noise with a moderate decay envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.randn(len(t))
    envelope = np.exp(-25 * t)
    clap = envelope * noise
    return clap

def generate_lead(freq, duration, sr, volume):
    """
    Generate a lead synth note using a sawtooth wave.
    A light envelope and clipping add a bit of distortion for a party edge.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    saw = 2 * (t * freq - np.floor(0.5 + t * freq))
    envelope = np.exp(-5 * t)
    lead = volume * envelope * saw
    lead = np.clip(lead, -0.8, 0.8)
    return lead

def generate_siren_effect(duration, sr):
    """
    Generate a siren effect: a sine wave that sweeps upward in frequency.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = np.linspace(200, 600, len(t))
    phase = 2 * np.pi * np.cumsum(freqs) / sr
    envelope = np.exp(-3 * t)
    siren = envelope * np.sin(phase)
    return siren

# --- Remix Beat Thread ---
def remix_beat_thread():
    """
    A 16-step sequencer that layers several drum sounds:
      - Kick on steps 0, 4, 8, 12.
      - Snare on steps 4 and 12.
      - Hi-hat on every even step.
      - Clap on step 8.
      - Synth hits on steps 0, 7, and 15.
    The combined mix is amplified and processed with an 8D stereo effect.
    """
    global current_beats, current_freq, current_volume, sample_rate
    step = 0
    while True:
        sixteenth = (60.0 / current_beats) / 4  # duration of a sixteenth note
        sounds = []
        if step % 16 in [0, 4, 8, 12]:
            kick = generate_kick(current_freq, 0.12, sample_rate)
            sounds.append(kick)
        if step % 16 in [4, 12]:
            snare = generate_snare(0.08, sample_rate)
            sounds.append(snare)
        if step % 2 == 0:
            hihat = generate_hihat(0.04, sample_rate)
            sounds.append(hihat)
        if step % 16 == 8:
            clap = generate_clap(0.08, sample_rate)
            sounds.append(clap)
        if step % 16 in [0, 7, 15]:
            synth = generate_synth(current_freq, 0.1, sample_rate, current_volume)
            sounds.append(synth)

        if sounds:
            max_len = max(len(s) for s in sounds)
            mix = np.zeros(max_len)
            for s in sounds:
                if len(s) < max_len:
                    s = np.pad(s, (0, max_len - len(s)), 'constant')
                mix += s
            if np.max(np.abs(mix)) > 0:
                mix = mix / np.max(np.abs(mix))
            mix = mix * 2.0  # Amplify for a loud party vibe
            mix_stereo = apply_8d_effect(mix, sample_rate, pan_speed=0.3)
            try:
                sd.play(mix_stereo, samplerate=sample_rate, blocking=True)
            except Exception as e:
                print("Error playing remix beat:", e)

        time.sleep(sixteenth)
        step = (step + 1) % 16

# --- Lead Thread ---
def lead_thread():
    """
    Every measure (4 beats) play a lead note using a sawtooth wave.
    The lead frequency is remapped to a party range and processed with 8D panning.
    """
    global current_freq, current_volume, current_beats, sample_rate
    while True:
        measure_duration = (60.0 / current_beats) * 4
        lead_duration = (60.0 / current_beats) * 0.5  # half-beat duration
        lead_freq = np.interp(current_freq, [200, 2000], [300, 800])
        lead_note = generate_lead(lead_freq, lead_duration, sample_rate, current_volume)
        lead_note = lead_note * 2.0
        lead_stereo = apply_8d_effect(lead_note, sample_rate, pan_speed=0.2)
        try:
            sd.play(lead_stereo, samplerate=sample_rate, blocking=True)
        except Exception as e:
            print("Error playing lead:", e)
        time.sleep(max(0, measure_duration - lead_duration))

# --- Party Thread ---
def party_thread():
    """
    Every 4 beats, trigger a random vocal-chop style noise burst,
    processed with 8D panning and high amplification.
    """
    global current_beats, sample_rate
    while True:
        time.sleep((60.0 / current_beats) * 4)
        duration = 0.2
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        noise = np.random.randn(len(t))
        envelope = np.exp(-30 * t)
        chop = noise * envelope
        chop = chop * 3.0  # Extra amplification
        chop_stereo = apply_8d_effect(chop, sample_rate, pan_speed=0.5)
        try:
            sd.play(chop_stereo, samplerate=sample_rate, blocking=True)
        except Exception as e:
            print("Error playing party chop:", e)

# --- Siren Thread ---
def siren_thread():
    """
    Every 16 beats, trigger a siren effect (sweeping tone) to add remix flavor.
    """
    global current_beats, sample_rate
    while True:
        time.sleep((60.0 / current_beats) * 16)
        duration = 0.5
        siren = generate_siren_effect(duration, sample_rate)
        siren = siren * 2.5  # Amplify
        siren_stereo = apply_8d_effect(siren, sample_rate, pan_speed=0.4)
        try:
            sd.play(siren_stereo, samplerate=sample_rate, blocking=True)
        except Exception as e:
            print("Error playing siren:", e)

# Start all audio threads as daemons.
threading.Thread(target=remix_beat_thread, daemon=True).start()
threading.Thread(target=lead_thread, daemon=True).start()
threading.Thread(target=party_thread, daemon=True).start()
threading.Thread(target=siren_thread, daemon=True).start()

# --- macOS System Volume Control ---
def set_system_volume(vol):
    """
    Set the macOS system output volume using AppleScript.
    (vol is 0 to 100)
    """
    os.system(f"osascript -e 'set volume output volume {int(vol)}'")

# --- MediaPipe & OpenCV Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Open the camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the frame
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand = None
    right_hand = None

    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            thumb_lm = handLms.landmark[4]
            index_lm = handLms.landmark[8]
            thumb_tip = (int(thumb_lm.x * w), int(thumb_lm.y * h))
            index_tip = (int(index_lm.x * w), int(index_lm.y * h))
            cv2.circle(frame, thumb_tip, 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, index_tip, 10, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            if hand_label == "Left":
                left_hand = {"thumb": thumb_tip, "index": index_tip}
            elif hand_label == "Right":
                right_hand = {"thumb": thumb_tip, "index": index_tip}

    # --- Compute Audio Parameters from Hand Gestures ---
    volume_val = 0   # (0-100)
    freq_val = 0     # in Hz
    beats_val = 0    # BPM

    # Volume: distance between both index fingertips
    if left_hand is not None and right_hand is not None:
        left_index = left_hand["index"]
        right_index = right_hand["index"]
        vol_distance = math.hypot(right_index[0] - left_index[0],
                                  right_index[1] - left_index[1])
        volume_val = np.interp(vol_distance, [50, w], [0, 100])
        cv2.line(frame, left_index, right_index, (0, 255, 0), 3)

    # Frequency: right-hand thumb-to-index distance
    if right_hand is not None:
        r_thumb = right_hand["thumb"]
        r_index = right_hand["index"]
        r_distance = math.hypot(r_index[0] - r_thumb[0],
                                r_index[1] - r_thumb[1])
        freq_val = np.interp(r_distance, [20, 200], [200, 2000])
        cv2.line(frame, r_thumb, r_index, (255, 0, 0), 3)

    # Beats (BPM): left-hand thumb-to-index distance
    if left_hand is not None:
        l_thumb = left_hand["thumb"]
        l_index = left_hand["index"]
        l_distance = math.hypot(l_index[0] - l_thumb[0],
                                l_index[1] - l_thumb[1])
        beats_val = np.interp(l_distance, [20, 200], [60, 180])
        cv2.line(frame, l_thumb, l_index, (0, 0, 255), 3)

    # Update global parameters.
    current_volume = volume_val / 100.0
    current_freq = freq_val if freq_val > 0 else 440.0
    current_beats = beats_val if beats_val > 0 else 120.0

    set_system_volume(volume_val)

    cv2.putText(frame, f'Volume: {int(volume_val)}%', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Freq: {int(freq_val)} Hz', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Beats: {int(beats_val)} BPM', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    num_lines = 5
    for i in range(num_lines):
        x_pos = int((i + 1) * w / (num_lines + 1))
        cv2.line(frame, (x_pos, 0), (x_pos, h), (255, 255, 255), 1)
        t_val = time.time()
        amplitude = int((volume_val / 100) * (h / 4))
        offset = h // 2
        dot_y = int(amplitude * math.sin(t_val * (freq_val / 500) + i) + offset)
        cv2.circle(frame, (x_pos, dot_y), 8, (0, 255, 255), cv2.FILLED)

    cv2.imshow("Party Remix Control", frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
