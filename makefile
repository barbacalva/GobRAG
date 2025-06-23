###############################################################################
#                          BOE RAG PIPELINE – Makefile                        #
#                                                                             #
#  Uso rápido                                                                 #
#  ----------                                                                 #
#  make            ⇒ ejecuta toda la cadena (target por defecto)              #
#  make -j8        ⇒ lo mismo, con 8 hilos en las fases paralelizables        #
#  make download   ⇒ sólo descarga las leyes                                  #
#  make parse      ⇒ convierte a Markdown + JSONL chunks                      #
#  make embed      ⇒ genera embeddings e inserta en Chroma                    #
#  make clean      ⇒ borra todos los artefactos intermedios                   #
###############################################################################

# ------------- VARIABLES BÁSICAS -------------------------------------------
PY          ?= python
SCRIPTS_DIR := ingest
DATA_DIR    := data

RAW_DIR     := $(DATA_DIR)/raw
CHUNK_DIR   := $(DATA_DIR)/chunks
MD_DIR      := $(DATA_DIR)/md
STORE_DIR   := $(DATA_DIR)/vector_store
COLLECTION  := boe_es_v1

LIST_XML    := $(DATA_DIR)/law_list.xml
RAW_INDEX   := $(RAW_DIR)/download_index.csv
CHUNK_INDEX := $(CHUNK_DIR)/chunk_index.csv

# --------- CENTINELAS (archivos vacíos que marcan cada fase terminada) ---
RAW_DONE   := $(RAW_DIR)/.done
CHUNK_DONE := $(CHUNK_DIR)/.done
STORE_DONE := $(STORE_DIR)/.done

# ------------- REGLA PRINCIPAL ---------------------------------------------
.PHONY: all
all: embed          ## Ejecuta la pipeline completa

# ------------- ETAPA 1 – CATÁLOGO ------------------------------------------
$(LIST_XML): $(SCRIPTS_DIR)/fetch_law_list.py
	@echo "==> Descargando lista de legislación consolidada"
	$(PY) $(SCRIPTS_DIR)/fetch_law_list.py \
			--output $(LIST_XML) --overwrite

.PHONY: list
list: $(LIST_XML)   ## Solo descarga el catálogo (law_list.xml)

# ------------- ETAPA 2 – DESCARGA DE LEYES ----------------------------------
$(RAW_DONE): $(LIST_XML) $(SCRIPTS_DIR)/fetch_laws.py
	@echo "==> Descargando leyes consolidadas"
	$(PY) $(SCRIPTS_DIR)/fetch_laws.py \
	       --input $(LIST_XML) \
	       --output-dir $(RAW_DIR) \
	       --index $(RAW_INDEX) \
	       --workers 16
	@touch $@

.PHONY: download
download: $(RAW_DONE)          ## Descarga todos los XML de leyes

# ------------- ETAPA 3 – CHUNKING ------------------------------------------
# Lista dinámica de XML presentes en la carpeta raw:
RAW_XML = $(wildcard $(RAW_DIR)/*.xml)
# Traduce a lista de JSONL
CHUNK_FILES = $(patsubst $(RAW_DIR)/%.xml,$(CHUNK_DIR)/%.jsonl,$(RAW_XML))

$(CHUNK_DIR)/%.jsonl: $(RAW_DIR)/%.xml $(SCRIPTS_DIR)/process_laws.py
	@echo "==> Parseando y troceando $*"
	$(PY) $(SCRIPTS_DIR)/process_laws.py \
	      --raw-dir $(RAW_DIR) \
	      --chunk-dir $(CHUNK_DIR) \
	      --md-dir $(MD_DIR) \
	      --max-tokens 512 --overlap 50 --workers 8

$(CHUNK_DONE): $(RAW_DONE) $(CHUNK_FILES)
	@touch $@

.PHONY: parse
parse: $(CHUNK_DONE)           ## Genera Markdown y chunks JSONL

# ------------- ETAPA 4 – EMBEDDINGS ----------------------------------------
$(STORE_DONE): $(CHUNK_DONE) $(SCRIPTS_DIR)/embed_chunks.py
	@echo "==> Generando embeddings e insertando en Chroma"
	$(PY) $(SCRIPTS_DIR)/embed_chunks.py \
        	--chunk-dir $(CHUNK_DIR) \
			--store $(STORE_DIR) \
			--collection $(COLLECTION) \
			--model jinaai/jina-embeddings-v2-base-es \
			--device mps \
			--batch 128 --skip-existing
	@touch $@

.PHONY: embed
embed: $(STORE_DONE)        ## Crea/actualiza la vector DB en Chroma

# ------------- LIMPIEZA -----------------------------------------------------
.PHONY: clean
clean:                          ## Elimina todos los artefactos
	@echo "==> Limpiando directorios data/"
	rm -rf $(RAW_DIR) $(CHUNK_DIR) $(MD_DIR) $(STORE_DIR) $(LIST_XML)