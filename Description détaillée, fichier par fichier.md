```
engine/
├─ __init__.py
├─ main.py
├─ api.py
├─ engine.py
├─ schemas.py
├─ config.py
├─ auth.py
├─ logging_conf.py
├─ requirements.txt
└─ Dockerfile
```

------

## Description détaillée, fichier par fichier

### `__init__.py`

| Élément                            | Rôle                       | Entrées | Sorties                                                      |
| ---------------------------------- | -------------------------- | ------- | ------------------------------------------------------------ |
| `__all__ = ["settings", "Engine"]` | Rend l’API interne propre. | —       | Exporte l’objet « settings » (singleton) et la classe `Engine`. |

------

### `config.py`

| Élément                        | Rôle                                                         | Entrées (env)                                                | Sorties            |
| ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ |
| `class Settings(BaseSettings)` | Centralise la config (modèle, device, quotas, chemin du prompt, clés API admises…). | `MODEL_ID`, `DEVICE`, `HF_TOKEN`, `MAX_TOKENS`, `PROMPT_PATH` **ou** `PROMPT_TEXT`, `ALLOWED_KEYS`, etc. | Instance immuable. |
| `settings = Settings()`        | Singleton accessible partout.                                | Env déjà chargée                                             | Objet `Settings`.  |

------

### `logging_conf.py`

| Élément               | Rôle                                                         | Entrées                | Sorties                     |
| --------------------- | ------------------------------------------------------------ | ---------------------- | --------------------------- |
| `configure_logging()` | Initialise `logging` (format JSON, rotation quotidienne, niveau via `LOG_LEVEL`). | `LOG_LEVEL`, `LOG_DIR` | Configure le logger racine. |

------

### `auth.py`

| Élément                                      | Rôle                                                         | Entrées                     | Sorties                              |
| -------------------------------------------- | ------------------------------------------------------------ | --------------------------- | ------------------------------------ |
| `verify_key(api_key)`                        | Vérifie la présence de l’API-key dans `settings.ALLOWED_KEYS`. | Chaîne `api_key`            | Booléen ou lève `HTTPException 401`. |
| `rate_limit(user_id, tokens)`                | Incrémente le compteur (en RAM ou Redis) et bloque si dépasse. | `user_id`, nombre de tokens | Lève `HTTPException 429` ou `None`.  |
| `current_user(request)` (dépendance FastAPI) | Extrait, valide et retourne l’id utilisateur.                | Header `Authorization`      | `user_id` str.                       |

------

### `schemas.py`

*Pydantic conforme à la spec OpenAI 2024-05-01*

| Modèle                                   | Rôle                           | Champs majeurs                                               |
| ---------------------------------------- | ------------------------------ | ------------------------------------------------------------ |
| `ChatCompletionRequest`                  | Requête `/v1/chat/completions` | `model`, `messages[]`, `max_tokens`, `temperature`, `stream`… |
| `ChatCompletionResponse`                 | Réponse non-stream             | `id`, `created`, `model`, `choices[]`, `usage`               |
| `ChatChunk`                              | Évènement SSE pour streaming   | `id`, `choices[]`, `delta`                                   |
| `EmbeddingRequest` / `EmbeddingResponse` | Routes embeddings              | —                                                            |
| `ModelCard`                              | Pour `/v1/models`              | `id`, `object`, `created`, `owned_by`                        |
| `Usage`                                  | Comptage tokens                | `prompt_tokens`, `completion_tokens`, `total_tokens`         |

------

### `engine.py`

| Élément                                       | Rôle                                                         | Entrées                              | Sorties                     |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------ | --------------------------- |
| `class Engine`                                | Portage GPU de MedBot. Conserve l’index RAG et l’historique. | `settings` à l’init                  | Voir méthodes ci-dessous    |
| `Engine.load()` *(@classmethod)*              | Charge tokenizer, modèle HF sur `settings.DEVICE`, initialise FAISS. | —                                    | Instance unique (singleton) |
| `add_docs(texts:list[str])` / `add_pdf(path)` | Alimente la base de connaissances.                           | Texte ou chemin PDF                  | nb passages ajoutés         |
| `_retrieve(query, k)`                         | Renvoie les *k* passages pertinents (RAG).                   | prompt user, `k`                     | `list[str]`                 |
| `generate_completion(messages, **hp)`         | Réponse complète non-stream.                                 | Liste messages OpenAI + hyper-params | (`str reply`, `Usage`)      |
| `stream_completion(messages, **hp)`           | Génère jeton par jeton (yield).                              | idem                                 | itérateur `str`             |
| `set_history(messages)`                       | Injecte un historique extérieur (restore conv).              | Liste messages                       | None                        |
| `fallback_attention()` *(static)*             | Patching Flash-Attention en cas d’erreur runtime.            | —                                    | None                        |

------

### `api.py`

| Route                       | Rôle                   | Corps/Requête                                     | Corps/Réponse                 |
| --------------------------- | ---------------------- | ------------------------------------------------- | ----------------------------- |
| `POST /v1/chat/completions` | Mode bloc              | `ChatCompletionRequest` JSON                      | `ChatCompletionResponse` JSON |
| idem `?stream=true`         | Streaming              | SSE : suite de `ChatChunk`, terminée par `[DONE]` | —                             |
| `POST /v1/embeddings`       | Embeddings vectoriels  | `EmbeddingRequest`                                | `EmbeddingResponse`           |
| `GET /v1/models`            | Lister modèles exposés | —                                                 | `[ModelCard]`                 |
| `GET /healthz`              | Probe K8s/compose      | —                                                 | `200 OK` + `"up"`             |

**Pipeline interne :**

1. Dépendance `current_user` → vérif clé + quota.
2. Conversion Pydantic → appel `engine.generate_completion` ou `stream_completion`.
3. Comptage tokens, remplissage `usage`.
4. Log JSON d’audit, puis retour réponse ou flux SSE.

------

### `main.py`

| Élément                                    | Rôle                                            | Entrées | Sorties               |
| ------------------------------------------ | ----------------------------------------------- | ------- | --------------------- |
| `app = FastAPI(...)`                       | Application centrale.                           | —       | Objet FastAPI         |
| **startup event**                          | Charge `Engine.load()` dans `app.state.engine`. | —       | None (modèle en VRAM) |
| **shutdown event**                         | Libère VRAM proprement.                         | —       | None                  |
| `include_router(api.router, prefix="/v1")` | Monte toutes les routes compatibles OpenAI.     | —       | —                     |

------

### `requirements.txt`

- `fastapi`, `uvicorn[standard]`,
- `pydantic>=2`,
- `torch==2.x` avec CUDA, `transformers`, `sentence-transformers`, `faiss-gpu`,
- `pillow`, `PyPDF2`, `python-dotenv`, `redis` *(option rate-limit)*.

------

### `Dockerfile`

*Base :* `pytorch/pytorch:2.2.0-cuda12.1-runtime`
 *Étapes :* copie source → install `requirements.txt` → expose `8000` → `CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]`.
 *Volumes conseillés :*

- `/root/.cache/huggingface` (poids modèle)
- `/data/` (pdf/doc)
- `/logs/` (audit JSON)

------

## Flux E/S global

1. **Requête LAN** (`POST /v1/chat/completions`) ⟶ `api.py`
2. `api` → vérif `auth.py` → charge messages → **`engine.generate_completion`**
3. Génération PyTorch ; `Usage` calculé.
4. Retour JSON (ou SSE) à l’appelant ; log écrit par `logging_conf`.

Cette fiche sert de référence : chaque fichier, classe et méthode y est répertorié avec son rôle, ses entrées et ses sorties.