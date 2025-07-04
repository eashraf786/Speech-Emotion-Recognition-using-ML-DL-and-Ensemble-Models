import tkinter as tk
from tkinter import filedialog, messagebox, Label, Scale
import numpy as np
import librosa,emoji
import pickle
import opensmile
import soundfile as sf
from keras.models import load_model
import warnings
import pygame
from PIL import Image, ImageTk
warnings.filterwarnings('ignore')

scaler_path = "D:/MTECH CSE/3rd Sem/PBL-2/GUI app/scalerLE_for_ann_model_val_acc_97.20_emodb_opensmile.pkl"
try:
    with open(scaler_path, "rb") as scaler_file:
        scaler, le = pickle.load(scaler_file)
except FileNotFoundError:
    raise FileNotFoundError("Ensure scalerLE_for_ann_model_val_acc_97.20_emodb_opensmile.pkl is present in the directory.")

mlp = load_model('D:/MTECH CSE/3rd Sem/PBL-2/GUI app/best_ann_model_val_acc_97.20_emodb_opensmile.keras')

pygame.init()  
pygame.mixer.init()
wid,hei = 1500,800
class SpeechEmotionRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Emotion Recognizer")
        self.root.geometry(f"{wid}x{hei}")
        self.root.configure(bg='black') 

        self.setup_background()
        
        self.current_file = None
        
        self.create_ui()
        
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        self.root.after(100, self.check_music_end)
        
    def setup_background(self):
        try:
            bg_image = Image.open("D:/MTECH CSE/3rd Sem/PBL-2/GUI app/background_image.jpg")
            
            bg_image = bg_image.resize((wid, hei), Image.LANCZOS)
            
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            
            self.canvas = tk.Canvas(self.root, width=900, height=600)
            self.canvas.pack(fill="both", expand=True)
            
            self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        except Exception as e:
            print(f"Could not load background image: {e}")
            self.canvas = tk.Canvas(self.root, width=wid, height=hei, highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)
            self.create_gradient_background()
        
    def create_gradient_background(self):
        for i in range(hei):
            r = int(max(0, 30 - i/20))
            g = int(max(0, 50 - i/12))
            b = int(max(0, 80 - i/8))
            color = f'#{r:02x}{g:02x}{b:02x}'
            self.canvas.create_line(0, i, wid, i, fill=color)
        
    def create_ui(self):
        content_frame = tk.Frame(self.canvas, bg='#14062b', bd=0, 
                                 highlightthickness=0, 
                                 relief='flat')
        content_frame.place(relx=0.5, rely=0.5, anchor='center', 
                            width=750, height=500)

        title_label = Label(content_frame, text="Speech Emotion Recognizer", 
                            font=("Arial", 24, "bold"), 
                            fg='white', bg='#14062b')
        title_label.pack(pady=(20,10))

        instruction_label = Label(content_frame, 
                                  text="Upload an audio file to detect emotional tone", 
                                  font=("Arial", 14), 
                                  fg='#8E8E9D', bg='#14062b')
        instruction_label.pack(pady=5)

        upload_button = tk.Button(
            content_frame, 
            text="Upload Audio", 
            command=self.classify_audio, 
            font=("Arial", 14, "bold"),
            bg='#4A4A6A', 
            fg='white',
            activebackground='#5A5A7A',
            relief=tk.FLAT,
            padx=20,
            pady=10
        )
        upload_button.pack(pady=10)

        self.result_label = Label(
            content_frame, 
            text="", 
            font=("Arial", 16, "bold"), 
            fg='#50FA7B',
            bg='#14062b'
        )
        self.result_label.pack(pady=10)

        audio_control_frame = tk.Frame(content_frame, bg='#14062b')
        audio_control_frame.pack(pady=10)

        self.play_pause_btn = tk.Button(
            audio_control_frame, 
            text="Play", 
            command=self.toggle_play_pause, 
            state=tk.DISABLED,
            font=("Arial", 12, "bold"),
            bg='#4A4A6A',
            fg='white',
            activebackground='#5A5A7A',
            relief=tk.FLAT,
            padx=15,
            pady=8
        )
        self.play_pause_btn.pack(side=tk.LEFT, padx=5)

        volume_container = tk.Frame(audio_control_frame, bg='#14062b')
        volume_container.pack(side=tk.LEFT, padx=10)

        self.volume_label = Label(
            volume_container, 
            text="Volume:", 
            font=("Arial", 12), 
            fg='white', 
            bg='#14062b'
        )
        self.volume_label.pack()

        self.volume_slider = Scale(
            volume_container, 
            from_=0, 
            to=100, 
            orient=tk.HORIZONTAL, 
            length=200, 
            command=self.adjust_volume,
            bg='#14062b',
            fg='white',
            highlightthickness=0,
            troughcolor='#4A4A6A',
            activebackground='#5A5A7A'
        )
        self.volume_slider.set(100) 
        self.volume_slider.pack()

    def preprocess_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=16000) 
            ytrim, _ = librosa.effects.trim(audio, top_db=25)
            smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            sf.write('trimmed_audio.wav', ytrim, sr)
            sml = smile.process_file('trimmed_audio.wav')
            sml.reset_index(drop=True, inplace=True)
            return sml
        except Exception as e:
            messagebox.showerror("Error", f"Error processing audio file: {e}")
            return None
    
    def classify_audio(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")]
        )
        if not file_path:
            return

        features = self.preprocess_audio(file_path)
        if features is None:
            return 
        features_scaled = scaler.transform(features)
        prediction = mlp.predict(features_scaled, verbose=0)
        predicted_labels = le.inverse_transform(np.argmax(prediction, axis=1))
        ejd = {'happy':':beaming_face_with_smiling_eyes:','sad':':pensive_face:','angry':':enraged_face:',
               'neutral':':neutral_face:','fear':':fearful_face:','boredom':':unamused_face:','disgust':':face_vomiting:'}
        emo = ejd[predicted_labels[0]]
        self.result_label.config(text=f'Predicted Emotion: {predicted_labels[0]} {emoji.emojize(emo)}')

        self.current_file = file_path
        pygame.mixer.music.load(self.current_file)
        self.play_pause_btn.config(state=tk.NORMAL, text="Play")

    def toggle_play_pause(self):
        if not self.current_file:
            return

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.pause()
            self.play_pause_btn.config(text="Play")
        else:
            pygame.mixer.music.play()
            self.play_pause_btn.config(text="Play")

    def adjust_volume(self, val):
        volume = int(val) / 100
        pygame.mixer.music.set_volume(volume)

    def check_music_end(self):
        try:
            for event in pygame.event.get():
                if event.type == pygame.USEREVENT:
                    self.play_pause_btn.config(text="Play")
        except Exception as e:
            print(f"Error checking music end: {e}")
            
        self.root.after(100, self.check_music_end)

root = tk.Tk()
app = SpeechEmotionRecognizer(root)
root.mainloop()