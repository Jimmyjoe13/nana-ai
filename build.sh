#!/bin/bash

# Fonction pour gérer les erreurs
handle_error() {
    echo "Une erreur est survenue dans la ligne : $1"
    exit 1
}

# Activer le mode erreur
set -e
trap 'handle_error $LINENO' ERR

echo " Démarrage du processus de build..."

# Installation des dépendances Python
echo " Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

# Création des répertoires nécessaires
echo " Création des répertoires..."
mkdir -p /opt/render/project/src/data

# Téléchargement et mise en cache du modèle
echo " Téléchargement du modèle..."
python -c "
try:
    from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
    import os

    model_name = 'facebook/blenderbot-400M-distill'
    cache_dir = '/opt/render/project/src/data/model'

    print(f'Téléchargement du tokenizer {model_name}...')
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    
    print(f'Téléchargement du modèle {model_name}...')
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    
    print('Sauvegarde du modèle et du tokenizer...')
    tokenizer.save_pretrained(cache_dir)
    model.save_pretrained(cache_dir)
    
    print(' Modèle et tokenizer sauvegardés avec succès!')
except Exception as e:
    print(f'Erreur lors du téléchargement du modèle: {str(e)}')
    exit(1)
"

echo " Build terminé avec succès!"
