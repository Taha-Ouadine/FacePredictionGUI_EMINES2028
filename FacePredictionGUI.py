import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import PIL.Image, PIL.ImageTk   
import numpy as np
import threading
import requests 
import time
import os
import sys
scriptpath = os.path.dirname(os.path.abspath(__file__))

from deepface import DeepFace
from datetime import datetime
from sklearn.cluster import KMeans
import tensorflow as tf
import numpy as np

# Test Models
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error

# import LoadUTK



# Display Landmarks
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def DisplayLandmarks(cv_image, colors=[(0,255,0),(255,0,0),(0,0,255)]):
    rgb_frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for i, landmarks in enumerate(results.multi_face_landmarks):
            color = colors[i%len(colors)]
            # Draw all 468 landmarks
            for idx, landmark in enumerate(landmarks.landmark):
                h, w, _ = cv_image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(cv_image, (x, y), 1, color, -1)
    final_image = cv_image.copy()
    return final_image




VIDEO_SLEEP = 0.03
KICKSTART_VIDEO = True
PREPROCESS_DEEPFACE = True

# region Class Initialization

class FaceAnalyzerApp:
    def __init__(self, window, window_title, ModelPath='Final__script/MODELS', DeepFaceImported=False):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        self.window.configure(bg="#f0f0f0")
        self.modelpath = ModelPath
        self.DeepFaceLoadable = DeepFaceImported
    
       
        # Modèles
        self.predictions = {
            'age': {
                'options': ['AgeGenderDeep','yu4u'] + (['DeepFace'] if DeepFaceImported else []),
                'model': tk.StringVar(value='DeepFace'),
                'active': tk.BooleanVar(value=True),
                'deepface_attr': 'age',
                'last_prediction': None
            },
            'gender': {
                'options': ['AgeGenderDeep'] + (['DeepFace'] if DeepFaceImported else []),
                'model': tk.StringVar(value='DeepFace'),
                'active': tk.BooleanVar(value=True),
                'deepface_attr': 'gender',
                'last_prediction': None
            },
            'ethnicity': {
                'options': ['VGG-Face'] + (['DeepFace'] if DeepFaceImported else []),
                'model': tk.StringVar(value='DeepFace'),
                'active': tk.BooleanVar(value=True),
                'deepface_attr': 'race',
                'last_prediction': None
            },
            'emotion': {
                'options': ['FER'] + (['DeepFace'] if DeepFaceImported else []),
                'model': tk.StringVar(value='FER'),
                'active': tk.BooleanVar(value=True),
                'deepface_attr': 'emotion',
                'last_prediction': None
            },
            'color': {
                'options': [], 'model': None,
                'active': tk.BooleanVar(value=False),
                'deepface_attr': None, 'last_prediction': None
            }
        }

        # Setup Models
        self.models = {
            'DeepFace': self.Predict_Deepface, 
            'Caffe': self.Predict_Caffe, 
            'FER': self.Predict_Fer,
            'AgeGenderDeep': self.Predict_AgeGenderDeep, 
            'VGG-Face': self.Predict_VGGFace, 
            'yu4u': self.Predict_Yu4uAge  # Ajout de yu4u
        }
        self.initialize_models(None if PREPROCESS_DEEPFACE else ['Caffe', 'AgeGenderDeep', 'VGG-Face', 'FER','yu4u'])
       
        # Variables pour les options d'affichage
        self.show_video = tk.BooleanVar(value=True) # Visualiser le contenu de la camera
        self.auto_stop_video = tk.BooleanVar(value=True) # S'arrêter pour visualiser les Predictions
        self.is_video_streaming = False
        self.cap = None
        self.video_thread = None
        self.stop_video = False
        self.current_image = None
       
        # Création de l'interface
        self.create_widgets()
        if KICKSTART_VIDEO: self.start_video(show_error=False)
       
        # Protocole de fermeture
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
    def Predict_Yu4uAge(self, img, toPredict):
        try:
            assert toPredict == "age"

            # Model file details
            yu4u_model_url = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/age_only_resnet50_weights.061-3.300-4.410.hdf5"
            yu4u_model_filename = "age_only_resnet50_weights.061-3.300-4.410.hdf5"
            model_local_path = os.path.join(self.modelpath, yu4u_model_filename)

            # Download if not present
            if not hasattr(self, 'yu4u_model'):
                if not os.path.exists(model_local_path):
                    print(f"[YU4U] Downloading model from {yu4u_model_url} ...")
                    os.makedirs(self.modelpath, exist_ok=True)
                    r = requests.get(yu4u_model_url)
                    if r.status_code == 200:
                        with open(model_local_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print("[YU4U] Model downloaded successfully.")
                    else:
                        print(f"[YU4U ERROR] Échec du téléchargement, code HTTP: {r.status_code}")
                        return {'age': None}, {'age': None}
                    print("[YU4U] Model downloaded.")
                    size = os.path.getsize(model_local_path)
                    print(f"[YU4U] Taille du modèle: {size} octets")

                # Load model
                from keras.models import load_model
                self.yu4u_model = load_model(model_local_path, compile=False)
                print("[YU4U] Model loaded.")

            # Image preprocessing
            from keras.utils import img_to_array
            from keras.applications.resnet50 import preprocess_input

            face_img = cv2.resize(img, (224, 224))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_array = img_to_array(face_img)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = preprocess_input(face_array)

            # Predict
            predicted_distribution = self.yu4u_model.predict(face_array)[0]
            predicted_age = int(np.round(np.sum(predicted_distribution * np.arange(0, 101))))
            print(f"[YU4U DEBUG] Âge prédit: {predicted_age}")

            result = {"age": predicted_age}

            # ✅ Important : retourne une liste de dicts
            return [result], [result]

        except Exception as e:
            print(f"[YU4U ERROR] Erreur lors de la prédiction d’âge: {str(e)}")
            empty = {"age": None}
            return [empty], [empty]

    def initialize_models(self, models=None):
        model_list = ['Caffe', 'AgeGenderDeep', 'DeepFace', 'VGG-Face', 'FER','yu4u']
        if models is None: models=model_list
        print("INITIALIZE MODELS:", models)
        for model_name in models:
            if model_name=='Caffe': continue
            if model_name=='AgeGenderDeep': continue
            if model_name=='DeepFace' and self.DeepFaceLoadable: self.Predict_Deepface(None, [])
            if model_name=='VGG-Face': continue
            if model_name=='FER': continue
            if model_name == 'yu4u':
                try:
                    # Teste le chargement du modèle YU4U avec une image test
                    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    result = self.Predict_Yu4uAge(test_img, "age")
                    print("YU4U model initialized successfully")
                except Exception as e:
                    print(f"Failed to initialize YU4U model: {str(e)}")
                    import traceback
                    traceback.print_exc()   
    def select_best_models(self, selected_keys=None):
        if selected_keys is None: selected_keys=list(self.predictions.keys())
        selected_keys = list(selected_keys)
        print(selected_keys)
        for key in list(self.predictions.keys()):
            if key in selected_keys and self.predictions[key]['model'] is not None: self.predictions[key]['model'].set('DeepFace')
   
    def delete_stored_results(self):
            self.results_text.delete(1.0, tk.END)
            for key in list(self.predictions.keys()): self.predictions[key]['last_prediction']=None
       

    # region WIDGETS MENUS
    #----------------------------------- WIDGETS AND MENU ------------------------------------
    def create_widgets(self):
        style = ttk.Style()

        # Création des éléments de style cyberpunk
        style.element_create('Cyberpunk.TLabelFrame.border', 'from', 'default')
        style.layout('Cyberpunk.TLabelFrame', [
            ('Cyberpunk.TLabelFrame.border', {'sticky': 'nswe', 'children': [
                ('Cyberpunk.TLabelFrame.padding', {'sticky': 'nswe', 'children': [
                    ('Cyberpunk.TLabelFrame.label', {'sticky': 'nswe'})
                ]})
            ]})
        ])

        # Configuration des couleurs UNIQUEMENT pour le style cyberpunk
        style.configure('Cyberpunk.TLabelFrame', 
                background='#1a1a2e',     # Bleu foncé pour les groupes
                foreground='#00d4ff',     # Cyan électrique pour les titres
                bordercolor='#00d4ff',    # Bordure cyan
                borderwidth=2,
                relief="solid",
                anchor="center",          # Centrer le texte
                font=('Arial', 10, 'bold'))

        style.configure('Cyberpunk.TLabelFrame.Label', 
                background='#1a1a2e', 
                foreground='#00d4ff')

        # Configuration du thème principal
        style.configure('TFrame', background="#0a0a0f")
        # Boutons avec style cyberpunk et néon
        style.configure('TButton', 
                    background='#16213e',     # Bleu marine foncé
                    foreground='#00d4ff',     # Cyan néon
                    borderwidth=2,
                    focuscolor='#ff006e',     # Rose néon au focus
                    padding=8)
        
        # Style spécial pour le bouton PRÉDIRE - Ultra flashy
        style.configure('Predict.TButton',
                    background='#ff006e',     # Rose néon
                    foreground='#ffffff',     # Blanc pur
                    borderwidth=3,
                    padding=10)
        
        # Labels avec texte cyan sur fond sombre
        style.configure('TLabel', 
                    background='#1a1a2e', 
                    foreground='#00d4ff')
        
        # Checkbuttons avec style néon COMPLET
        style.configure('TCheckbutton', 
                    background='#1a1a2e',
                    foreground='#00d4ff',
                    focuscolor='#ff006e',
                    indicatorcolor='#16213e',      # Couleur de la case à cocher
                    selectcolor='#ff006e')         # Couleur quand cochée
        
        # Scrollbar cyberpunk style
        style.configure('Vertical.TScrollbar',
                    background='#16213e',          # Fond de la scrollbar
                    troughcolor='#0a0a0f',         # Couleur du rail
                    bordercolor='#00d4ff',         # Bordure
                    arrowcolor='#00d4ff',          # Couleur des flèches
                    darkcolor='#1a1a2e',
                    lightcolor='#1a1a2e')
        
        # Configuration des états hover/active - Effets néon
        style.map('TButton',
                background=[('active', '#0f3460'),    # Bleu plus vif au survol
                            ('pressed', '#ff006e')],   # Rose néon quand pressé
                foreground=[('active', '#ffffff'),
                            ('pressed', '#ffffff')])
        
        style.map('Predict.TButton',
                background=[('active', '#ff1a7a'),    # Rose plus vif au survol
                            ('pressed', '#00d4ff')])   # Cyan quand pressé
        
        style.map('TCheckbutton',
                background=[('active', '#1a1a2e'),
                            ('pressed', '#1a1a2e')],
                foreground=[('active', '#ff006e')])
        
        style.map('Vertical.TScrollbar',
                background=[('active', '#ff006e')])
        
        main_frame = ttk.Frame(self.window, style='TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Zone Affichage Image
        self.display_frame = ttk.LabelFrame(main_frame, text="Affichage", style="Cyberpunk.TLabelFrame")
        self.display_frame.configure(width=640, height=480)
        self.display_frame.pack_propagate(False)

        self.display_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.canvas = tk.Canvas(self.display_frame, width=640, height=480, 
                            bg="#000011",           # Bleu très sombre pour le canvas
                            highlightbackground="#00d4ff",  # Bordure cyan néon
                            highlightthickness=2)
        self.canvas.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.display_id = None

        # Zone de Contrôles
        controls_frame = ttk.LabelFrame(main_frame, text="Contrôles", style="Cyberpunk.TLabelFrame")
        controls_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        # Zone de choix des caractéristiques à prédire
        prediction_frame = ttk.LabelFrame(controls_frame, text="Caractéristiques à prédire", style="Cyberpunk.TLabelFrame")
        prediction_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)

        ttk.Checkbutton(prediction_frame, text="Âge", variable=self.predictions['age']['active']).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(prediction_frame, text="Sexe", variable=self.predictions['gender']['active']).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(prediction_frame, text="Ethnicité", variable=self.predictions['ethnicity']['active']).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(prediction_frame, text="Émotion", variable=self.predictions['emotion']['active']).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(prediction_frame, text="Couleur dominante", variable=self.predictions['color']['active']).pack(anchor=tk.W, padx=5)

        # Options d'affichage
        display_frame = ttk.LabelFrame(controls_frame, text="Options d'affichage", style="Cyberpunk.TLabelFrame")
        display_frame.pack(fill=tk.X, padx=5, pady=5, ipady=5)

        ttk.Checkbutton(display_frame, text="Afficher la vidéo", variable=self.show_video).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(display_frame, text="Arrêter pour visualiser", variable=self.auto_stop_video).pack(anchor=tk.W, padx=5)

        # Boutons d'actions
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(action_frame, text="Démarrer Caméra", command=self.start_video).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(action_frame, text="Arrêter Caméra", command=self.stop_video_stream).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(action_frame, text="Charger Image", command=self.load_image).pack(side=tk.LEFT, padx=5, pady=5)

        # Bouton de prédiction avec style spécial
        predict_button = ttk.Button(controls_frame, text="PRÉDIRE", 
                                command=self.predict, 
                                style='Predict.TButton')
        predict_button.pack(fill=tk.X, padx=5, pady=10)

        # Affichage des résultats avec couleurs néon ultra cool
        self.result_display = {'Age':[None,'#ff3366'], 'Gender':[None,'#33ff99'], 'Ethnicity':[None,'#ffaa00'],
                            'Emotion':[None,'#cc66ff'], 'Dominant Color':[None,'#00d4ff',None]}
        
        for key in list(self.result_display.keys()):
            this_row = tk.Frame(controls_frame, bg='#1a1a2e')
            this_row.pack(fill=tk.X, padx=5, pady=2)
            
            # Label avec couleur néon personnalisée sur fond sombre
            label = tk.Label(this_row, text=f"{str(key)}: ", 
                             bg='#1a1a2e',
                            foreground=self.result_display[key][1],
                            background='#1a1a2e',
                            font=('Arial', 9, 'bold'),
                             borderwidth=0)
            label.pack(side=tk.LEFT)
            
            self.result_display[key][0] = tk.StringVar()
            self.result_display[key][0].set("")
            
            # Label de résultat avec texte cyan néon
            result_label = tk.Label(this_row, textvariable=self.result_display[key][0],
                                foreground='#00d4ff',
                                background='#1a1a2e',
                                font=('Arial', 9))
            result_label.pack(side=tk.LEFT)
            
            if key=='Dominant Color':
                # FINI LE BLANC ! Maintenant c'est cyberpunk aussi !
                self.result_display[key][2] = tk.Label(this_row, 
                                                    bg="#16213e",        # Fond bleu marine au lieu de blanc
                                                    width=2, height=1, 
                                                    relief="solid", bd=2,
                                                    highlightbackground="#ff006e",
                                                    highlightthickness=2)
                self.result_display[key][2].pack(side=tk.LEFT, padx=5)
        self.window.option_add('*Menu*background', '#1a1a2e')  # Force le fond des menus
        self.window.option_add('*Menu*foreground', '#00d4ff')
        self.window.option_add('*Menu*activeBackground', '#ff006e')
        self.window.option_add('*Menu*activeForeground', '#ffffff')
        # Zone de détails de résultats
        results_frame = ttk.LabelFrame(main_frame, text="Résultats En Détail", style="Cyberpunk.TLabelFrame")
        results_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        # Text widget avec thème cyberpunk
        self.results_text = tk.Text(results_frame, height=10, width=40, wrap=tk.WORD,
                                bg="#0a0a0f",           # Bleu très sombre
                                fg="#00d4ff",           # Texte cyan néon
                                insertbackground="#ff006e",  # Curseur rose néon
                                selectbackground="#16213e",  # Sélection bleu marine
                                selectforeground="#ffffff",  # Texte sélectionné blanc
                                font=('Consolas', 10),
                                borderwidth=2,
                                relief="solid",
                                highlightbackground="#00d4ff",
                                highlightcolor="#ff006e",
                                highlightthickness=2)
        self.results_text.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Barre de défilement CYBERPUNK !
        scrollbar = ttk.Scrollbar(self.results_text, command=self.results_text.yview,
                                style='Vertical.TScrollbar')
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)

        # Configuration pour le redimensionnement
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Configuration de la fenêtre principale avec thème cyberpunk
        self.window.configure(bg='#0a0a0f')
        self.create_menu()
        
    def create_menu(self):
        menu_bar = tk.Menu(self.window, 
                        bg='#1a1a2e',           # Fond bleu foncé
                        fg='#00d4ff',           # Texte cyan
                        activebackground='#ff006e',  # Fond rose au survol
                        activeforeground='#ffffff',  # Texte blanc au survol
                        selectcolor='#00d4ff',       # Couleur de sélection
                        borderwidth=2,
                        relief='solid')

        # Menu Fichier
        file_menu = tk.Menu(menu_bar, tearoff=0,
                        bg='#1a1a2e', fg='#00d4ff',
                        activebackground='#ff006e', activeforeground='#ffffff',
                        selectcolor='#00d4ff',
                        borderwidth=1)
        file_menu.add_command(label="Ouvrir Image", command=self.load_image)
        file_menu.add_command(label="Sauvegarder Résultats", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.on_closing)
        menu_bar.add_cascade(label="Fichier", menu=file_menu)

        # Menu Modèles
        models_menu = tk.Menu(menu_bar, tearoff=0,
                            bg='#1a1a2e', fg='#00d4ff',
                            activebackground='#ff006e', activeforeground='#ffffff',
                            selectcolor='#00d4ff')

        age_menu = tk.Menu(models_menu, tearoff=0,
                        bg='#1a1a2e', fg='#00d4ff',
                        activebackground='#ff006e', activeforeground='#ffffff',
                        selectcolor='#00d4ff')
        if self.DeepFaceLoadable:
            age_menu.add_radiobutton(label="DeepFace", variable=self.predictions['age']['model'], value="DeepFace")
        age_menu.add_radiobutton(label="AgeGenderDeep", variable=self.predictions['age']['model'], value="AgeGenderDeep")
        age_menu.add_radiobutton(label="Caffe", variable=self.predictions['age']['model'], value="Caffe")
        age_menu.add_radiobutton(label="YU4U", variable=self.predictions['age']['model'], value = "yu4u" )
        
        gender_menu = tk.Menu(models_menu, tearoff=0,
                            bg='#1a1a2e', fg='#00d4ff',
                            activebackground='#ff006e', activeforeground='#ffffff',
                            selectcolor='#00d4ff')
        if self.DeepFaceLoadable:
            gender_menu.add_radiobutton(label="DeepFace", variable=self.predictions['gender']['model'], value="DeepFace")
        gender_menu.add_radiobutton(label="AgeGenderDeep", variable=self.predictions['gender']['model'], value="AgeGenderDeep")

        ethnicity_menu = tk.Menu(models_menu, tearoff=0,
                                bg='#1a1a2e', fg='#00d4ff',
                                activebackground='#ff006e', activeforeground='#ffffff',
                                selectcolor='#00d4ff')
        if self.DeepFaceLoadable:
            ethnicity_menu.add_radiobutton(label="DeepFace", variable=self.predictions['ethnicity']['model'], value="DeepFace")
        ethnicity_menu.add_radiobutton(label="VGG-Face", variable=self.predictions['ethnicity']['model'], value="VGG-Face")

        emotion_menu = tk.Menu(models_menu, tearoff=0,
                            bg='#1a1a2e', fg='#00d4ff',
                            activebackground='#ff006e', activeforeground='#ffffff',
                            selectcolor='#00d4ff')
        if self.DeepFaceLoadable:
            emotion_menu.add_radiobutton(label="DeepFace", variable=self.predictions['emotion']['model'], value="DeepFace")
        emotion_menu.add_radiobutton(label="FER", variable=self.predictions['emotion']['model'], value="FER")

        color_menu = None

        models_menu.add_cascade(label="Modèle d'Âge", menu=age_menu)
        models_menu.add_cascade(label="Modèle de Sexe", menu=gender_menu)
        models_menu.add_cascade(label="Modèle d'Ethnicité", menu=ethnicity_menu)
        models_menu.add_cascade(label="Modèle d'Émotion", menu=emotion_menu)
        if color_menu is not None: models_menu.add_cascade(label="Modèle de Couleur", menu=color_menu)
        menu_bar.add_cascade(label="Modèles", menu=models_menu)

        # Menu Aide
        help_menu = tk.Menu(menu_bar, tearoff=0,
                        bg='#1a1a2e', fg='#00d4ff',
                        activebackground='#ff006e', activeforeground='#ffffff',
                        selectcolor='#00d4ff')
        help_menu.add_command(label="Guide d'Utilisation", command=self.show_guide)
        help_menu.add_command(label="À Propos des Modèles", command=self.show_models_info)
        help_menu.add_command(label="À Propos", command=self.show_about)
        menu_bar.add_cascade(label="Aide", menu=help_menu)

        # Auto Select Best Model
        menu_bar.add_command(label="Autoselect-Models", command=self.select_best_models)

        # Test Model With Utk
        utk_test_menu = tk.Menu(menu_bar, tearoff=0,
                            bg='#1a1a2e', fg='#00d4ff',
                            activebackground='#ff006e', activeforeground='#ffffff',
                            selectcolor='#00d4ff')
        # Select UTK Folder (BUTTON)
        utk_test_menu.add_command(label="Select Folder", command=self.select_utk_folder)
        # Select Model To Test (SUBMENU)
        utktest_model_menu = tk.Menu(utk_test_menu, tearoff=0,
                                    bg='#1a1a2e', fg='#00d4ff',
                                    activebackground='#ff006e', activeforeground='#ffffff',
                                    selectcolor='#00d4ff')
        for modelname in self.models.keys():
            utktest_model_menu.add_command(label=modelname, command=lambda modelname=modelname:self.test_with_utk(modelname))
        utk_test_menu.add_cascade(label="Test Model", menu=utktest_model_menu)
        menu_bar.add_cascade(label="Test With UTK", menu=utk_test_menu)
        self.utk_folder_path = "E:/tahao_mw5iph3/Desktop/Python/# AI/UTKFace"

        self.window.config(menu=menu_bar)

    # region CAMERA DISPLAY
    #----------------------------------- CAMERA AND IMAGE DISPLAY ------------------------------------

   
    def start_video(self, show_error=True):
        if self.is_video_streaming:
            return
       
        self.stop_video = False
        self.cap = cv2.VideoCapture(0)
       
        if not self.cap.isOpened():
            if show_error: messagebox.showerror("Erreur", "Impossible d'ouvrir la caméra")
            return
       
        self.is_video_streaming = True
        self.video_thread = threading.Thread(target=self.video_stream)
        self.video_thread.daemon = True
        self.video_thread.start()
   
    def video_stream(self):
        while not self.stop_video:
            ret, frame = self.cap.read()
            if not ret:
                break
               
            # Convertir l'image pour l'affichage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
            # Afficher l'image si l'option est activée
            if self.show_video.get():
                self.current_image = frame.copy()
                self.display_image(frame)
           
            time.sleep(VIDEO_SLEEP)  # Limiter la fréquence d'images
       
        # Libérer la caméra
        if self.cap:
            self.cap.release()
       
        self.is_video_streaming = False
   
    def stop_video_stream(self):
        self.stop_video = True
        if self.video_thread:
            self.video_thread.join(timeout=1.0)
       
        if self.cap:
            self.cap.release()
            self.cap = None
        self.display_image(None)        
        self.is_video_streaming = False
   
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner une image",
            filetypes=[("Images", ".jpg;.jpeg;.png;.bmp")]
        )
       
        if not file_path:
            return
       
        try:
            # Charger l'image avec OpenCV
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("L'image n'a pas pu être chargée")
           
            # Convertir l'image pour l'affichage
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.current_image = img.copy()
           
            # Afficher l'image
            self.display_image(img, lock=True, source="LOAD")
           
            # Effacer les anciens résultats
            self.delete_stored_results()
           
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement de l'image: {str(e)}")
   
    def display_image(self, img, lock=False, source="CAMERA"):
        # Redimensionner l'image pour l'adapter au canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
       
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 640
            canvas_height = 480
       
        if img is None and self.display_id is not None:
            self.canvas.delete(self.display_id)
            return
       
        # Calculer les dimensions pour garder le ratio
        h, w = img.shape[:2]
        ratio = min(canvas_width / w, canvas_height / h)
        new_size = (int(w * ratio), int(h * ratio))
       
        # Redimensionner l'image
        resized_img = cv2.resize(img, new_size)

        # Modifier l'image
        final_image = self.DisplayModifications(resized_img)
       
        # Convertir en format PIL
        pil_img = PIL.Image.fromarray(final_image)
       
        # Convertir en format PhotoImage
        photo_img = PIL.ImageTk.PhotoImage(image=pil_img)
       
        # Mettre à jour le canvas
        if lock:
            self.show_video.set(False)
            time.sleep(VIDEO_SLEEP)
        if source=="CAMERA" and self.show_video.get()==False: return # Mesure de sécurité contre l'écrasement
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.display_id = self.canvas.create_image(new_size[0] // 2, new_size[1] // 2, image=photo_img)
        self.canvas.image = photo_img  # Garder une référence


    def DisplayModifications(self, cv_image):
        cv_image = DisplayLandmarks(cv_image)
        final_image = cv_image.copy()
        return final_image
   

    # region MODELS
    #----------------------------------- MODEL PREDICTION LOGIC ------------------------------------

   
    def get_dominant_color(self, face_region):
        """Extrait la couleur dominante d'une région faciale"""
        try:
            pixels = face_region.reshape(-1, 3)
            kmeans = KMeans(n_clusters=1, n_init=10)
            kmeans.fit(pixels)
            color = kmeans.cluster_centers_[0]
            return tuple(map(int, color))  # (R, G, B)
        except: return None
   
    def Predict_Caffe(self, img, toPredict):
        return [], []
   
    def Predict_VGGFace(self, img, toPredict):
        return [], []
   
    def Predict_AgeGenderDeep(self, img, toPredict):
        """AgeGenderDeep"""
        try:
            assert toPredict in ["age", "gender"]
            # Modèle simple utilisant OpenCV pour la démonstration
            # Dans une application réelle, vous devriez charger un modèle AgeNet spécifique
            # Ceci est une simplification à des fins de démonstration
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
           
            results = []
            for (x, y, w, h) in faces:
                face_region = {'x': x, 'y': y, 'w': w, 'h': h}
                # Simulation d'une prédiction (dans une application réelle, utilisez un vrai modèle)
                # Calculez l'âge moyen à partir des pixels pour la démonstration
                face_img = gray[y:y+h, x:x+w]
                mean_pixel = np.mean(face_img)
                simulated_age = int(mean_pixel / 255 * 60 + 10)  # Simule un âge entre 10 et 70
               
                result = {
                    'region': face_region,
                    f'{toPredict}': simulated_age
                }
                results.append(result)
               
            return results, results
        except Exception as e:
            print(f"Erreur dans predict_with_agegenderdeep ({toPredict}): {e}")
            return [], []
           
    def Predict_Fer(self, img, toPredict):
        """Prédit l'émotion en utilisant un modèle FER"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
           
            results = []
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
           
            for (x, y, w, h) in faces:
                face_region = {'x': x, 'y': y, 'w': w, 'h': h}
               
                # Simulation de prédiction (dans une application réelle, utilisez un vrai modèle)
                face_img = gray[y:y+h, x:x+w]
                mean_pixel = np.mean(face_img)
               
                # Créer des probabilités simulées pour les émotions
                base_probs = np.random.random(len(emotions)) * 10
                # Favorise une émotion en fonction de la valeur moyenne des pixels
                emotion_index = int(mean_pixel / 255 * len(emotions))
                base_probs[emotion_index % len(emotions)] += 30
               
                # Normaliser pour que la somme soit 100
                total = sum(base_probs)
                emotion_probs = {emotion: (prob / total) * 100 for emotion, prob in zip(emotions, base_probs)}
               
                result = {
                    'region': face_region,
                    'emotion': emotion_probs
                }
                results.append(result)
               
            return results, results
        except Exception as e:
            print(f"Erreur dans predict_emotion_with_fer: {e}")
            return [], []
       
    def Predict_Deepface(self, img, attributes):
        if attributes is None or not isinstance(attributes,list) or len(attributes)==0:
            empty_img = np.zeros((480, 640, 3), dtype=np.uint8) # Blank prediction to initialize the model
            results = DeepFace.analyze(empty_img, actions=['age','gender','race','emotion'], enforce_detection=False, detector_backend='opencv')
            return [], []
        if img is None:
            return [], []
        results = DeepFace.analyze(
            img,
            actions=attributes,
            enforce_detection=False,
            detector_backend='opencv'
        )

        # Format output
        result_dir = { key:None for key in list(self.predictions.keys()) }
        processed_results = []
        for result in results:
            key_list = list(result.keys())
            result_dir['age'] = result['age'] if 'age' in key_list else None
            result_dir['gender'] = result['dominant_gender'] if 'dominant_gender' in key_list else None
            result_dir['ethnicity'] = result['race'] if 'race' in key_list else None
            result_dir['emotion'] = result['emotion'] if 'emotion' in key_list else None
            processed_results.append(result_dir)
        assert len(results)==len(processed_results)
       
        return results, processed_results
   

    # region PREDICT BUTTON
    #----------------------------------- PREDICTION MAIN FUNCTION ------------------------------------#

   
    def predict(self):
        if self.current_image is None:
            messagebox.showinfo("Information", "Aucune image à analyser. Veuillez charger une image ou démarrer la caméra.")
            return
       
        try:
            # Copier l'image pour éviter les modifications pendant l'analyse
            img_copy = self.current_image.copy()
           
            # Convertir en BGR pour DeepFace
            img_bgr = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
           
            # POUR L'INSTANT, LE CODE NE SUPPORTE QU'UN SEUL VISAGE A LA FOIS !!!
            model_keys = list(self.predictions.keys())
            result_dict = { key:None for key in model_keys+['region'] } # Note: Keep 'region' last
            results=[]
            DeepfaceAttributes = None if not self.DeepFaceLoadable else []

            # Au moins une caractéristique doit être sélectionnée
            if not any([self.predictions[key]['active'].get()==True for key in model_keys]):
                messagebox.showinfo("Information", "Veuillez sélectionner au moins une caractéristique à prédire.")
                return
           
            print("--------------- PREDICTION START ---------------")
           
            # Sélection du modèle selon le choix de l'utilisateur
            for prediction_name in model_keys:
                prediction_info = self.predictions[prediction_name]
                if len(prediction_info['options'])==0: continue # No model available for this prediction

                if not prediction_info['active'].get()==True: continue
                model = prediction_info['model'].get()
                print(f"[DEBUG] Prédiction {prediction_name} avec modèle {model}")
                if model not in prediction_info['options']:
                    messagebox.showinfo("Erreur", "Un modèle non disponible a été sélectionné.")
                    continue
                predictions = [{'empty':None}]
                if model != 'DeepFace':
                    print(f"[DEBUG] Appelant {model} pour {prediction_name}")
                    predictions, processed_predictions = self.models[model](img_copy, prediction_name)
                    print(f"[DEBUG] Résultat de {model}: {processed_predictions}")
                    if (processed_predictions is not None) and (len(processed_predictions)>0) and (result_dict[prediction_name] is None):
                        result_dict[prediction_name]=processed_predictions[0][prediction_name]
                        print("RESULT DICT SO FAR:", result_dict)
                elif self.DeepFaceLoadable and prediction_info['deepface_attr'] is not None:
                    DeepfaceAttributes.append(prediction_info['deepface_attr'])
                    print("DEEPFACE ATTRIBUTE APPENDED:", prediction_info['deepface_attr'])
                if result_dict['region'] is None and len(predictions)>0 and 'region' in list(predictions[0].keys()):
                    result_dict['region'] = predictions[0]['region'] # Fill region key with the first available data
                    print("ADDED REGION")

            if DeepfaceAttributes is not None and len(DeepfaceAttributes)>0:
                raw_deepface_results, processed_deepface_results = self.Predict_Deepface(img_bgr, DeepfaceAttributes)
                deepface_result = processed_deepface_results[0] # LE PREMIER VISAGE UNIQUEMENT
                for key in list(deepface_result.keys()):
                    if key in list(result_dict.keys()) and result_dict[key] is None:
                        result_dict[key] = deepface_result[key]
                    elif key=='race':
                        result_dict['ethnicity'] = deepface_result[key]
                if result_dict['region'] is None and 'region' in list(raw_deepface_results[0].keys()):
                    result_dict['region'] = raw_deepface_results[0]['region']
                    print("ADDED REGION")

            # COLOR
            color_prediction_info = self.predictions['color']
            if color_prediction_info['active'].get()==True and result_dict['region'] is not None:
                try:
                    region = result_dict['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']
                    cropped_face = img_copy[y:y+h, x:x+w]
                    result_dict['color'] = self.get_dominant_color(cropped_face)
                except: messagebox.showinfo("Erreur", "La couleur n'a pas pu être prédite.")
           
            print("RESULT DICT SO FAR:", result_dict)

            results.append(result_dict)

            print("--------------- RESULTS ---------------")
            
            for key in list(result_dict.keys()):
                print(key, ":")
                print(result_dict[key])
           
            # Afficher les résultats
            if results:
                self.display_results(results, img_copy)
            else:
                messagebox.showinfo("Information", "Aucun visage détecté dans l'image.")
           
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'analyse: {str(e)}")
   
    # region PREDICT DISPLAY
    #----------------------------------- DISPLAY PREDICTIONS ------------------------------------

    def display_results(self, results, img):
        # Effacer les anciens résultats
        self.delete_stored_results()
       
        # Si aucun visage n'est détecté
        if not results:
            self.results_text.insert(tk.END, "Aucun visage détecté")
            return
       
        prediction_keys = list(self.predictions.keys())
        active_prediction_keys = [key for key in prediction_keys if self.predictions[key]['active'].get()==True]

        model_names = {key:self.predictions[key]['model'].get() for key in prediction_keys if key!='color'}
           
        # Afficher les résultats pour chaque visage détecté
        for i, face_result in enumerate(results):
            self.results_text.insert(tk.END, f"--------- Visage {i+1} ---------\n\n")

            this_result_keys = [key for key in list(face_result.keys())
                                if key in active_prediction_keys or key=='region']
           
            if 'age' in this_result_keys:
                age_value = face_result['age']
                self.result_display['Age'][0].set(str(age_value) if age_value is not None else 'None')
                model_name = model_names['age']
                self.results_text.insert(tk.END, f"Âge estimé: {age_value} (Modèle {model_name})\n")
           
            if 'gender' in this_result_keys:
                gender = face_result['gender']
                self.result_display['Gender'][0].set(str(gender) if gender is not None else 'None')
                model_name = model_names['gender']
                self.results_text.insert(tk.END, f"Sexe: {gender} (Modèle {model_name})\n")
           
            if 'ethnicity' in this_result_keys:
                ethnicity_results = face_result['ethnicity']
                model_name = model_names['ethnicity']
                self.results_text.insert(tk.END, f"Ethnicité (Modèle {model_name}):\n")
                if ethnicity_results is None:
                    self.result_display['Ethnicity'][0].set('None')
                    self.results_text.insert(tk.END, f"None\n")
                else:
                    # Trier les résultats par probabilité
                    sorted_ethnicities = sorted(ethnicity_results.items(), key=lambda x: x[1], reverse=True)
                    ethnicity_to_display = f"{str(sorted_ethnicities[0][0])} ({round(sorted_ethnicities[0][1],1)} %)"
                    if len(sorted_ethnicities)>=2:
                        ethnicity_to_display += f" {str(sorted_ethnicities[1][0])} ({round(sorted_ethnicities[1][1],1)} %)"
                    self.result_display['Ethnicity'][0].set(ethnicity_to_display)
                   
                    for ethnicity, prob in sorted_ethnicities:
                        self.results_text.insert(tk.END, f"  - {ethnicity}: {prob:.2f}%\n")
           
            if 'emotion' in this_result_keys:
                emotion_results = face_result['emotion']
                model_name = model_names['emotion']
                self.results_text.insert(tk.END, f"Émotion (Modèle {model_name}):\n")
               
                # Trier les résultats par probabilité
                if emotion_results is not None:
                    print(emotion_results)
                    sorted_emotions = sorted(emotion_results.items(), key=lambda x: x[1], reverse=True)
                    emotions_to_display = f"{str(sorted_emotions[0][0])} ({round(sorted_emotions[0][1],1)} %)"
                    if len(sorted_emotions)>=2:
                        emotions_to_display += f" {str(sorted_emotions[1][0])} ({round(sorted_emotions[1][1],1)} %)"
                    self.result_display['Emotion'][0].set(emotions_to_display)
                   
                    for emotion, prob in sorted_emotions:
                        self.results_text.insert(tk.END, f"  - {emotion}: {prob:.2f}%\n")
                else: self.result_display['Emotion'][0].set('None')

            if 'color' in this_result_keys:
                color_value = face_result['color']
                self.result_display['Dominant Color'][0].set(str(color_value) if color_value is not None else 'None')
                hex_color = '#%02x%02x%02x' % color_value
                self.result_display['Dominant Color'][2].config(bg=hex_color)
                self.result_display['Dominant Color'][2].pack(side=tk.LEFT, padx=5)
                self.results_text.insert(tk.END, f"Couleur Dominante: {color_value}\n")
           
            # Dessiner les rectangles sur l'image
            if 'region' in this_result_keys:
                img_with_boxes = self.draw_face_boxes(img, [face_result['region']])
                self.display_image(img_with_boxes, lock=self.auto_stop_video.get(), source="PREDICTION")
       
            # Sauvegarder le dernier résultat (CAR IL Y A UN SEUL VISAGE)
            for key in list(self.predictions.keys()): self.predictions[key]['last_prediction']=face_result[key]
   

    def draw_face_boxes(self, img, regions):
        img_with_boxes = img.copy()
        for region in regions:
            if region is None: continue
            box_color = (0, 255, 0) # RGB
            cv2.rectangle(img_with_boxes, (region['x'], region['y']), (region['x']+region['w'], region['y']+region['h']), box_color, 2)
        return img_with_boxes
   
    # region SAVE LOAD
    #----------------------------------- SAVING AND LOADING ------------------------------------

    def save_results(self):
        if not any([self.predictions[key]['active'].get()==True for key in list(self.predictions.keys())]):
            messagebox.showinfo("Information", "Aucun résultat à sauvegarder.")
            return
       
        try:
            # Proposer un nom de fichier
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Fichiers texte", "*.txt")],
                initialfile=f"analyse_faciale_{now}.txt"
            )
           
            if not file_path:
                return
           
            # Récupérer le texte des résultats
            results_text = self.results_text.get(1.0, tk.END)
           
            # Écrire dans le fichier
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"Résultats d'analyse faciale - {now}\n\n")
                f.write(results_text)
           
            messagebox.showinfo("Succès", f"Résultats sauvegardés dans {file_path}")
           
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")


    def select_utk_folder(self):
        print("- Select UTK Folder -")
        folder_path = filedialog.askdirectory( title="Où Est UTK Face ?" )
        if not folder_path:
            print("INVALID FOLDER PATH")
            return
        self.utk_folder_path = folder_path
        print(f"SELECTED {self.utk_folder_path}")

    
    def test_with_utk(self, model):
        print(f"Test Model {model} with UTKFace (Folder [{self.utk_folder_path}])")
        def test_thread():
            """UTKDataset, UTKLoader = LoadUTK.START_LOADING()
            convert_back = False
            if model=="DeepFace":
                predict_function = lambda img, attr=["age","gender"]: self.models[model](img,attr)
                convert_back = True # Convert Tensor back to image
            else: predict_function = self.models[model]
            testResult = LoadUTK.evaluate_model_generic(predict_function, UTKLoader, convert_back, print_progress=True, batch_count=2)"""
            def show_plot():
                fig_age, ax_age = plt.subplots()
                fig_gender, ax_gender = plt.subplots()
                #age_diff, gender_diff = testResult["Age DIFF"], testResult["Gender DIFF"]
                age_diff=np.array([-17,-21,29,-3,-15,-22,6,-12,7,-13,-1,3,41,3,1,-7,-6,21,-1])
                gender_diff=np.array([-1,0,-1,0,-1,0,0,-1,-1,0,-1,-1,0,-1,0,-1,-1,0,-1,-1])
                ax_age.hist(list(age_diff))
                # Genders: Male=0, Female=1. Diff=Real-Prediction.
                categories = {'Correct Guesses':list(gender_diff).count(0),
                              'False Males':list(gender_diff).count(1),
                              'False Females':list(gender_diff).count(-1)}
                ax_gender.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', startangle=90)
                ax_gender.axis('equal')
                fig_gender.tight_layout()
                fig_gender.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                fig_gender.set_size_inches(5, 5)
                # Launch Dialog
                self.show_plot_dialog("Error Histiogram: Age - Gender", [fig_age, fig_gender])
            self.window.after(0, show_plot)
        thread = threading.Thread(target=test_thread, daemon=True)
        thread.start()

    def show_plot_dialog(self, title, fig_list):
        # Créer une nouvelle fenêtre
        dialog = tk.Toplevel(self.window)
        dialog.title(title)
        dialog.geometry("800x600")
        dialog.resizable(True, True)

        # Créer le canvas matplotlib et l'insérer dans la fenêtre
        for fig in fig_list:
            canvas = FigureCanvasTkAgg(fig, master=dialog)
            canvas.draw()
            canvas.get_tk_widget().pack(side="left", fill=tk.BOTH, expand=True)

        # Bouton de fermeture
        close_button = ttk.Button(dialog, text="Fermer", command=dialog.destroy)
        close_button.pack(pady=10)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)

        # Rendre modal
        dialog.transient(self.window)
        dialog.grab_set()
        self.window.wait_window(dialog)
   

    # region HELP
    #----------------------------------- HELP INFO ------------------------------------
   

    def show_guide(self):
        guide_text = """
GUIDE D'UTILISATION
       
Cette application vous permet d'analyser des visages pour détecter des caractéristiques comme l'âge, le sexe, l'ethnicité et l'émotion.
       
COMMENT UTILISER L'APPLICATION:
       
1. SOURCE D'IMAGE
    - Démarrer la caméra: Utilise votre webcam pour capturer des images en temps réel.
    - Charger une image: Sélectionne une image depuis votre ordinateur.
       
2. OPTIONS D'ANALYSE
    - Cochez les caractéristiques que vous souhaitez analyser (âge, sexe, ethnicité, émotion).
    - Vous pouvez également activer ou désactiver l'affichage de la vidéo.
       
3. ANALYSE
    - Cliquez sur "PRÉDIRE" pour lancer l'analyse du visage avec les options sélectionnées.
    - Les résultats s'afficheront dans la zone de résultats à droite.
       
4. SAUVEGARDE
    - Vous pouvez sauvegarder les résultats en utilisant l'option "Sauvegarder Résultats" dans le menu Fichier.
       
5. MODÈLES
    - Différents modèles peuvent être sélectionnés dans le menu "Modèles" pour chaque caractéristique.
"""
       
        self.show_info_dialog("Guide d'Utilisation", guide_text)
   
    def show_models_info(self):
        models_info = """
INFORMATIONS SUR LES MODÈLES
       
Cette application utilise plusieurs modèles d'analyse faciale:
       
DEEPFACE:
    - Framework d'analyse faciale en Python
    - Offre des fonctionnalités de reconnaissance faciale, détection d'âge, de sexe, d'ethnicité et d'émotion
    - Utilise des réseaux de neurones pré-entraînés pour ses analyses
       
AGEGENDERDEEP:
    - Modèle spécialisé dans la détection de l'âge et du sexe
    - Basé sur des réseaux de neurones profonds (CNN)
    - Généralement plus précis pour les groupes d'âge que pour l'âge exact
       
CAFFE:
    - Modèle développé par Berkeley AI Research
    - Implémenté avec le framework Caffe
    - Connu pour sa bonne performance dans la détection d'âge
       
FER (FACIAL EMOTION RECOGNITION):
    - Modèle spécialisé dans la détection d'émotions faciales
    - Entraîné sur le dataset FER-2013 (Facial Expression Recognition)
    - Peut reconnaître 7 émotions de base: colère, dégoût, peur, joie, tristesse, surprise et neutralité
       
VGG-FACE:
    - Modèle basé sur les architectures VGG pour la reconnaissance faciale
    - Développé par le Visual Geometry Group de l'Université d'Oxford
    """
        self.show_info_dialog("À Propos des Modèles", models_info)

    def show_about(self):
        about_text = """
APPLICATION D'ANALYSE FACIALE
       
Version 1.0

Cette application permet de détecter et d'analyser des caractéristiques faciales à partir d'images ou de vidéos en temps réel.

Fonctionnalités:
    - Détection d'âge
    - Détection de sexe
    - Détection d'ethnicité
    - Détection d'émotion
    - Analyse en temps réel ou à partir d'images

Développée avec Python, Tkinter et DeepFace.
        """
       
        self.show_info_dialog("À Propos", about_text)
   
    def show_info_dialog(self, title, text):
        # Créer un dialogue
        dialog = tk.Toplevel(self.window)
        dialog.title(title)
        dialog.geometry("600x400")
        dialog.resizable(True, True)
       
        # Ajouter un widget Text
        text_widget = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
       
        # Ajouter une barre de défilement
        scrollbar = ttk.Scrollbar(text_widget, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
       
        # Insérer le texte
        text_widget.insert(tk.END, text)
        text_widget.config(state=tk.DISABLED)  # Rendre le texte en lecture seule
       
        # Bouton de fermeture
        close_button = ttk.Button(dialog, text="Fermer", command=dialog.destroy)
        close_button.pack(pady=10)
       
        # Rendre le dialogue modal
        dialog.transient(self.window)
        dialog.grab_set()
        self.window.wait_window(dialog)
   
    def on_closing(self):
        # Arrêter la vidéo si elle est en cours
        if self.is_video_streaming:
            self.stop_video_stream()
       
        # Fermer la fenêtre
        self.window.destroy()


# Fonction principale pour démarrer l'application
def main():
    root = tk.Tk()
    try:
        # Vérifier si DeepFace est installé
        import importlib
        importlib.import_module('deepface')
        # Créer l'application
        app = FaceAnalyzerApp(root, "Application d'Analyse Faciale", ModelPath=os.path.join(scriptpath,'MODELS'), DeepFaceImported=True)
    except ImportError:
        print("La bibliothèque DeepFace n'est pas installée. Elle ne sera pas utilisable cette session.")
        app = FaceAnalyzerApp(root, "Application d'Analyse Faciale", ModelPath=os.path.join(scriptpath,'MODELS'), DeepFaceImported=False)
    except Exception as e:
        print(f"Erreur: {str(e)}")
        sys.exit()
    root.mainloop()


if __name__ == "__main__":
    main()
