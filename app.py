import os
from dotenv import load_dotenv
import logging
import sys
import json
from datetime import datetime
import torch
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from heyoo import WhatsApp
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

app = FastAPI(title="NANA AI - Assistant WhatsApp")

# Vérification des variables d'environnement requises
required_env_vars = ['WHATSAPP_TOKEN', 'WHATSAPP_PHONE_ID', 'VERIFY_TOKEN']
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Variable d'environnement manquante : {var}")
        raise ValueError(f"Variable d'environnement manquante : {var}")

# Initialisation du client WhatsApp
try:
    messenger = WhatsApp(
        token=os.getenv('WHATSAPP_TOKEN'),
        phone_number_id=os.getenv('WHATSAPP_PHONE_ID')
    )
    logger.info("Client WhatsApp initialisé avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation du client WhatsApp: {e}")
    raise

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

# Initialisation du modèle Hugging Face
try:
    # Vérifier si le modèle est déjà téléchargé localement
    if os.path.exists(MODEL_PATH):
        logger.info(f"Chargement du modèle depuis {MODEL_PATH}")
        tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_PATH)
        model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_PATH)
    else:
        logger.info(f"Téléchargement du modèle {MODEL_NAME}")
        if os.getenv('HUGGINGFACE_TOKEN'):
            tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME, use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))
            model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME, use_auth_token=os.getenv('HUGGINGFACE_TOKEN'))
        else:
            tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
            model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        # Sauvegarder le modèle localement
        os.makedirs(MODEL_PATH, exist_ok=True)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    
    # Déplacer le modèle sur GPU si disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Modèle chargé avec succès sur {device}")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle Hugging Face: {e}")
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

# Initialisation de la mémoire des conversations
conversation_memory = ConversationMemory()

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

async def generate_response_with_model(user_input: str, sender_id: str) -> str:
    """Génère une réponse en utilisant le modèle Blenderbot avec gestion de la mémoire."""
    try:
        # Récupérer l'historique si activé
        context = ""
        if ENABLE_MEMORY:
            context = conversation_memory.get_conversation_history(sender_id)
            input_text = f"{context} {user_input}" if context else user_input
        else:
            input_text = user_input

        # Encoder l'entrée
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = inputs.to(device)

        # Générer la réponse
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_LENGTH,
                min_length=MIN_LENGTH,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                top_p=TOP_P,
                no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                do_sample=True
            )

        # Décoder la réponse
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Sauvegarder dans l'historique si activé
        if ENABLE_MEMORY:
            conversation_memory.add_message(sender_id, user_input, is_bot=False)
            conversation_memory.add_message(sender_id, response, is_bot=True)
        
        logger.info(f"Réponse générée pour {sender_id}: {response}")
        return response

    except Exception as e:
        logger.error(f"Erreur lors de la génération de réponse: {e}")
        return "Je suis désolé, j'ai du mal à générer une réponse pour le moment. Pourriez-vous reformuler votre message ?"

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Vérifie le webhook pour WhatsApp."""
    try:
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge", "")

        if mode and token:
            if mode == "subscribe" and token == os.getenv("VERIFY_TOKEN"):
                logger.info("Webhook vérifié avec succès")
                return challenge
            else:
                logger.error("Échec de la vérification du webhook")
                raise HTTPException(status_code=403, detail="Vérification du webhook échouée")
        else:
            logger.error("Paramètres manquants dans la requête webhook")
            raise HTTPException(status_code=400, detail="Paramètres manquants")
            
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du webhook: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook")
async def receive_message(request: Request):
    """Reçoit et traite les messages WhatsApp."""
    try:
        data = await request.json()
        logger.debug(f"Données reçues du webhook: {json.dumps(data, indent=2)}")
        
        if "object" not in data:
            logger.warning("Données reçues invalides - 'object' manquant")
            return {"status": "error", "message": "Invalid data format"}
        
        if data["object"] == "whatsapp_business_account":
            for entry in data["entry"]:
                for change in entry["changes"]:
                    if change["value"].get("messages"):
                        message_data = change["value"]["messages"][0]
                        sender_id = message_data["from"]
                        message_text = message_data["text"]["body"]

                        logger.info(f"Message reçu de {sender_id}: {message_text}")

                        message = Message(message=message_text, sender=sender_id)

                        # Vérification des messages urgents
                        if est_message_urgent(message.message):
                            logger.info("Message urgent détecté")
                            notifier_proprietaire(message.message, message.sender)
                            reponse = "J'ai bien reçu votre message urgent et je l'ai transmis au propriétaire. Il vous répondra dès que possible."
                        else:
                            logger.info("Message non urgent, génération de réponse avec le modèle")
                            reponse = await generate_response_with_model(message.message, sender_id)

                        logger.info(f"Envoi de la réponse à {sender_id}: {reponse}")
                        
                        messenger.send_message(
                            message=reponse,
                            recipient_id=sender_id
                        )
                        logger.info(f"Réponse envoyée avec succès à {sender_id}")

        return {"status": "success"}
    except Exception as e:
        logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
