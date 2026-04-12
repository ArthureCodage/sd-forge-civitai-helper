# 🐘 SD Forge — CivitAI Helper

Extension pour **Stable Diffusion Forge** permettant de télécharger,
scanner et mettre à jour vos modèles depuis CivitAI.

## Installation

```bash
cd <racine_SD_Forge>/extensions/
git clone <url_du_repo> sd-forge-civitai-helper
# Relancez Forge — install.py s'exécute automatiquement
```

Ou via l'interface : **Extensions → Install from URL**.

## Fonctionnalités

| Onglet | Fonctionnalité |
|---|---|
| ⬇️ Télécharger | Téléchargement par URL/ID, sélection version/fichier, resume, vérification SHA256, preview |
| 🔍 Recherche | Recherche par mot-clé, filtre par type, pagination |
| 🔄 Scan & MAJ | Scan SHA256 des modèles locaux, génération `.civitai.info`, détection de nouvelles versions |

## Clé API

Optionnelle pour les modèles publics.
Obligatoire pour les modèles restreints ou NSFW.

Obtenez-la sur : https://civitai.com/user/account

Vous pouvez aussi la définir en variable d'environnement :
```bash
export CIVITAI_API_KEY=votre_clé
```

## Structure des fichiers générés

Pour chaque modèle scanné, l'extension crée :
- `nom_du_modele.civitai.info` — métadonnées JSON
- `nom_du_modele.preview.jpg` — image de preview
