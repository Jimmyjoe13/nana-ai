# NANA AI - Assistant WhatsApp Intelligent

## Description
NANA AI est un assistant intelligent pour WhatsApp qui gère automatiquement vos conversations en votre absence. Il utilise l'API WhatsApp Business et le modèle Blenderbot de Hugging Face pour fournir des réponses contextuelles et naturelles en français.

## Fonctionnalités
- Réponse automatique aux messages WhatsApp
- Détection des messages urgents
- Notification au propriétaire pour les messages importants
- Intégration avec Blenderbot pour des réponses intelligentes
- Gestion du contexte des conversations
- Interface en français

## Prérequis
- Python 3.9.12 ou supérieur
- Compte WhatsApp Business
- Accès à l'API WhatsApp Business
- Compte Hugging Face avec token d'accès

## Installation

1. Clonez le dépôt :
```bash
git clone [URL_DU_REPO]
cd whatsapp-assistant
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurez les variables d'environnement :
- Copiez le fichier `.env.example` vers `.env`
- Remplissez les variables suivantes :
  ```env
  WHATSAPP_TOKEN=votre_token
  WHATSAPP_PHONE_ID=votre_phone_id
  WHATSAPP_BUSINESS_ID=votre_business_id
  VERIFY_TOKEN=votre_token_verification
  HUGGINGFACE_TOKEN=votre_token_huggingface
  ```

## Démarrage
```bash
uvicorn app:app --reload
```

## Déploiement sur Render

1. Créez un compte sur [Render](https://render.com)
2. Connectez votre dépôt GitHub à Render
3. Créez un nouveau Web Service
4. Configurez les variables d'environnement dans Render
5. Déployez l'application

## Configuration du Webhook WhatsApp
1. Une fois l'application déployée, récupérez l'URL Render
2. Configurez le webhook WhatsApp avec l'URL : `https://votre-app.onrender.com/webhook`
3. Utilisez le même VERIFY_TOKEN que celui configuré dans les variables d'environnement

## Personnalisation
- Les mots-clés urgents peuvent être modifiés dans le fichier `app.py`
- Les paramètres du modèle peuvent être ajustés dans le fichier `.env`

## Structure du Projet
```
whatsapp-assistant/
├── .env                # Variables d'environnement
├── .gitignore         # Fichiers à ignorer par Git
├── README.md          # Documentation
├── app.py            # Application principale
├── build.sh          # Script de build pour Render
├── render.yaml       # Configuration Render
└── requirements.txt  # Dépendances Python
```

## Sécurité
- Ne partagez jamais vos tokens d'API
- Utilisez toujours HTTPS en production
- Gardez vos dépendances à jour

## Support
Pour toute question ou problème, veuillez ouvrir une issue dans le dépôt GitHub.

## Licence
Ce projet est sous licence MIT.
