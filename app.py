import os
import sys
import logging
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from dotenv import load_dotenv
import torch
from heyoo import WhatsApp
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Configuration du logger
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Chargement des variables d'environnement
load_dotenv()
logger.info("Variables d'environnement chargées")

app = FastAPI(title="NANA AI - Assistant WhatsApp")

# Variables globales pour le modèle et le tokenizer
model = None
tokenizer = None
messenger = None

# Configuration des paramètres du modèle depuis les variables d'environnement
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/blenderbot-400M-distill')
MODEL_PATH = '/opt/render/project/src/data/model'
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 1000))
MIN_LENGTH = int(os.getenv('MIN_LENGTH', 50))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
TOP_K = int(os.getenv('TOP_K', 50))
TOP_P = float(os.getenv('TOP_P', 0.9))
NO_REPEAT_NGRAM_SIZE = int(os.getenv('NO_REPEAT_NGRAM_SIZE', 3))
NUM_RETURN_SEQUENCES = int(os.getenv('NUM_RETURN_SEQUENCES', 1))
ENABLE_MEMORY = os.getenv('ENABLE_MEMORY', 'True').lower() == 'true'
MAX_MEMORY_MESSAGES = int(os.getenv('MAX_MEMORY_MESSAGES', 10))
NUM_BEAMS = int(os.getenv('NUM_BEAMS', 4))
LENGTH_PENALTY = float(os.getenv('LENGTH_PENALTY', 0.6))

# Liste restreinte des mots-clés urgents
MOTS_CLES_URGENTS = [
    "URGENT", "URGENCE", "SOS",
    "CRITIQUE", "EMERGENCY"
]

def est_message_urgent(message: str) -> bool:
    """Vérifie si le message contient des mots-clés urgents."""
    message_upper = message.upper()
    is_urgent = any(mot in message_upper for mot in MOTS_CLES_URGENTS)
    logger.info(f"Vérification message urgent: '{message}' -> {is_urgent}")
    return is_urgent

def notifier_proprietaire(message: str, sender: str):
    """Envoie une notification au propriétaire pour les messages urgents."""
    try:
        logger.info(f"Tentative d'envoi de notification pour message urgent de {sender}")
        messenger.send_message(
            f"🚨 Message urgent de {sender}:\n{message}",
            os.getenv('WHATSAPP_PHONE_ID')
        )
        logger.info("Notification envoyée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la notification du propriétaire: {e}")
        raise

def init_whatsapp():
    """Initialise le client WhatsApp."""
    global messenger
    try:
        token = os.getenv("WHATSAPP_TOKEN")
        phone_id = os.getenv("WHATSAPP_PHONE_ID")
        messenger = WhatsApp(token, phone_number_id=phone_id)
        logger.info("Client WhatsApp initialisé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du client WhatsApp: {str(e)}")
        raise

def init_model():
    """Initialise le modèle et le tokenizer."""
    global model, tokenizer
    try:
        if model is None or tokenizer is None:
            logger.info("Chargement du modèle depuis le cache...")
            cache_dir = "/opt/render/project/src/data/model"
            
            logger.info("Chargement du tokenizer...")
            tokenizer = BlenderbotTokenizer.from_pretrained(cache_dir)
            
            logger.info("Chargement du modèle...")
            model = BlenderbotForConditionalGeneration.from_pretrained(cache_dir)
            
            # Configuration des tokens spéciaux
            if not hasattr(model.config, "decoder_start_token_id"):
                model.config.decoder_start_token_id = tokenizer.bos_token_id
            if not hasattr(model.config, "bos_token_id"):
                model.config.bos_token_id = tokenizer.bos_token_id
            if not hasattr(model.config, "eos_token_id"):
                model.config.eos_token_id = tokenizer.eos_token_id
            if not hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            logger.info(f"Modèle chargé avec succès sur {device}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        raise

class Message(BaseModel):
    message: str
    sender: str

# Structure pour stocker l'historique des conversations
class ConversationMemory:
    def __init__(self, max_messages=MAX_MEMORY_MESSAGES):
        self.conversations = {}
        self.max_messages = max_messages

    def add_message(self, sender_id: str, message: str, is_bot: bool = False):
        if sender_id not in self.conversations:
            self.conversations[sender_id] = []
        
        self.conversations[sender_id].append({
            'content': message,
            'is_bot': is_bot,
            'timestamp': datetime.now().isoformat()
        })
        
        # Garder seulement les derniers messages
        if len(self.conversations[sender_id]) > self.max_messages:
            self.conversations[sender_id] = self.conversations[sender_id][-self.max_messages:]

    def get_conversation_history(self, sender_id: str) -> str:
        if sender_id not in self.conversations:
            return ""
        
        history = []
        for msg in self.conversations[sender_id]:
            prefix = "Assistant: " if msg['is_bot'] else "Humain: "
            history.append(f"{prefix}{msg['content']}")
        
        return " ".join(history)

    def add_to_conversation(self, sender_id: str, user_input: str, response: str):
        self.add_message(sender_id, user_input, is_bot=False)
        self.add_message(sender_id, response, is_bot=True)

# Initialisation de la mémoire des conversations
conversation_memory = ConversationMemory()

async def generate_response_with_model(user_input: str, sender_id: str) -> str:
    """Génère une réponse en utilisant le modèle Blenderbot."""
    try:
        logger.info(f"Génération de réponse pour '{user_input}' de {sender_id}")
        
        # Vérification du modèle
        if model is None:
            logger.info("Modèle non chargé, chargement...")
            init_model()
            logger.info("Modèle chargé avec succès")
        
        logger.info(f"État du modèle - Device: {model.device}, Training: {model.training}")

        # Encoder l'entrée
        logger.info("Tokenization de l'entrée...")
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=128)
        logger.info(f"Tokens d'entrée: {inputs['input_ids'].tolist()}")
        inputs = inputs.to(model.device)
        logger.info("Tokenization terminée")

        # Générer la réponse
        logger.info("Génération de la réponse...")
        try:
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_length=128,
                    min_length=10,
                    num_beams=4,
                    do_sample=False,
                    early_stopping=True
                )
                logger.info(f"Tokens générés: {outputs.tolist()}")
        except Exception as gen_error:
            logger.error(f"Erreur pendant la génération: {str(gen_error)}", exc_info=True)
            raise

        # Décoder la réponse
        logger.info("Décodage de la réponse...")
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Réponse décodée: {response}")

        return response

    except Exception as e:
        logger.error(f"Erreur lors de la génération de réponse: {str(e)}", exc_info=True)
        return "Je suis désolé, j'ai du mal à comprendre. Pourriez-vous reformuler votre message ?"

@app.on_event("startup")
async def startup_event():
    """Événement de démarrage de l'application."""
    try:
        logger.info("🚀 Démarrage de l'application...")
        
        # Initialisation du client WhatsApp
        init_whatsapp()
        logger.info("Client WhatsApp initialisé avec succès")
        
        # Préchargement du modèle
        logger.info("🤖 Préchargement du modèle en arrière-plan...")
        init_model()
        
        logger.info("✅ Application démarrée avec succès")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Événement d'arrêt de l'application."""
    try:
        logger.info("🛑 Arrêt gracieux de l'application...")
        # Nettoyage des ressources si nécessaire
        logger.info("✅ Arrêt réussi")
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'arrêt: {str(e)}")
        raise

@app.get("/")
async def root():
    """Route racine pour vérifier que l'application fonctionne."""
    logger.info("Route racine appelée")
    return {"status": "ok", "message": "NANA AI is running!"}

@app.get("/ping")
async def ping():
    """Route de test simple."""
    logger.info("Route ping appelée")
    return {"ping": "pong"}

@app.get("/test")
async def test():
    """Route de test avec plus d'informations."""
    logger.info("Route test appelée")
    return {
        "status": "ok",
        "time": datetime.now().isoformat(),
        "env_vars": {
            "PORT": os.getenv("PORT"),
            "PYTHON_VERSION": os.getenv("PYTHON_VERSION"),
            "VERIFY_TOKEN": "***" if os.getenv("VERIFY_TOKEN") else None
        }
    }

@app.get("/load-model")
async def load_model():
    """Route pour charger le modèle manuellement."""
    logger.info("Chargement manuel du modèle...")
    try:
        init_model()
        return {"status": "ok", "message": "Modèle chargé avec succès"}
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Vérifie le webhook pour WhatsApp."""
    try:
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge", "")

        logger.info(f"Webhook verification - Mode: {mode}, Token: {token}, Challenge: {challenge}")

        if mode and token:
            if mode == "subscribe" and token == os.getenv("VERIFY_TOKEN"):
                logger.info("Webhook verified successfully")
                return int(challenge) if challenge.isdigit() else challenge
            else:
                logger.error(f"Webhook verification failed - Invalid mode or token")
                raise HTTPException(status_code=403, detail="Webhook verification failed")
        else:
            logger.error("Missing parameters in webhook request")
            raise HTTPException(status_code=400, detail="Missing parameters")
            
    except Exception as e:
        logger.error(f"Error during webhook verification: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def webhook_handler(request: Request):
    """Gère les webhooks entrants de WhatsApp."""
    try:
        body = await request.json()
        logger.info(f"Webhook reçu: {body}")
        
        if "object" not in body:
            logger.warning("Données reçues invalides - 'object' manquant")
            return {"status": "error", "message": "Invalid data format"}
        
        if body["object"] == "whatsapp_business_account":
            for entry in body["entry"]:
                for change in entry["changes"]:
                    if change["value"].get("messages"):
                        message_data = change["value"]["messages"][0]
                        if message_data.get("type") == "text":
                            # Extraction des informations du message
                            sender_id = message_data["from"]
                            message_text = message_data["text"]["body"]
                            
                            logger.info(f"Message reçu de {sender_id}: {message_text}")
                            
                            # Vérification si le message est urgent
                            if est_message_urgent(message_text):
                                logger.info("Message urgent détecté")
                                notifier_proprietaire(message_text, sender_id)
                                response = "J'ai bien reçu votre message urgent. Le propriétaire a été notifié et vous répondra dès que possible."
                            else:
                                logger.info("Message non urgent, génération de réponse avec le modèle")
                                try:
                                    # S'assurer que le modèle est chargé
                                    if model is None:
                                        logger.info("Chargement du modèle...")
                                        init_model()
                                        logger.info("Modèle chargé avec succès")
                                    
                                    # Générer la réponse
                                    response = await generate_response_with_model(message_text, sender_id)
                                    logger.info(f"Réponse générée: {response}")
                                    
                                    # Envoyer la réponse via WhatsApp
                                    logger.info(f"Envoi de la réponse à {sender_id}")
                                    messenger.send_message(response, sender_id)
                                    logger.info("Réponse envoyée avec succès")
                                    
                                except Exception as e:
                                    logger.error(f"Erreur lors de la génération/envoi de la réponse: {str(e)}")
                                    response = "Je suis désolé, j'ai rencontré une erreur. Pourriez-vous réessayer dans quelques instants ?"
                                    messenger.send_message(response, sender_id)
                            
                            return {"status": "ok", "message": "Message processed"}
        
        return {"status": "ok", "message": "Webhook received"}
    except Exception as e:
        logger.error(f"Erreur lors du traitement du webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
