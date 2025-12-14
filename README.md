# AI Folder — Segmentation & Training (Neodata)

Demo site: https://neodata.onrender.com/

Ova mapa (`ai/`) sadrži sve artefakte i skripte vezane uz semantičku segmentaciju, pripremu podataka i ručno treniranje modela koje koristimo u demonstraciji.

VAŽNO — Colab + ručni outputi
- Zbog ograničenja lokalnog hardvera, ključni korak segmentacije i izrade maski je izvršen ručno u Google Colab okruženju (GPU). Koristili smo `SAM_Improved_Segmentation.ipynb` za pokretanje SAM (SAM3) + YOLO pre-obradu i spremanje *segmented outputs* u Drive.
- Za demonstraciju (frontend i evaluaciju) koristimo upravo te spremljene izlaze (`segmented_output_improved/`) umjesto da svaki put ponovo pokrećemo dugotrajnu obradu.

Zašto to radimo
- Segmentacija s pixel-level maskama i treniranje SAM/YOLO pipeline-a zahtijeva snažnu GPU instancu i puno vremena. Colab + Drive omogućuje reproducibilnost bez potrebe za lokalnim high-end GPU-ima.

Sadržaj važnih datoteka
- `SAM_Improved_Segmentation.ipynb` — Glavni notebook koji:
  - pokreće YOLO da izreže fasadne cropove
  - u Colab-u pokreće SAM3 (text-prompt driven segmentation) nad cropovima
  - finira maske (morfologija, filtriranje, dedupliranje)
  - stvara overlay slike, konture i JSON metapodatke
  - sprema rezultate u `segmented_output_improved/` (struktura `positive/` i `negative/`)
- `neodata2/combined_detector.py` — Logika koja kombinira detekcije i računa metrike (koristi mask JSON i v2 classification rezultate)
- `classification_results_v2.json` — (ako postoji) mapira JPEG → CAD model (korišteno u pipeline-u)
- `segmented_output_improved/` — finalni izlazi iz notebooka: overlay slike, contour slike i `_masks.json` datoteke

Brzi vodič — pokretanje notebooka (Colab)
1. Otvori `SAM_Improved_Segmentation.ipynb` u Colab-u (File → Upload notebook or open from Drive).
2. Poveži Google Drive (notebook već sadrži `drive.mount('/content/drive')`).
3. Postavi varijable na vrhu (putanja do TRAIN podataka, `OUTPUT_DIR`, `YOLO_WEIGHTS`) — u notebooku su komentirane varijable `DRIVE_PATH`, `OUTPUT_DIR`, `YOLO_WEIGHTS`.
4. Pokreni ćelije redom. Notebook instalira potrebne pakete (`pillow-heif`, `ultralytics`, `decord` ili `av`) i učitava SAM3 model.
5. Nakon završetka, provjeri `OUTPUT_DIR` u Drive-u — tamo će se nalaziti `segmented_output_improved/` s podacima koji se koriste za demo.

Savjeti i napomene
- Ako Colab nema dovoljno RAM-a za SAM3, pokušaj koristiti Colab Pro / Pro+ ili prilagoditi `max_dim` u notebooku kako bi smanjio veličinu ulaznih slika.
- Notebook koristi razne varijante tekstualnih promptova (`COMPONENT_PROMPTS`) kako bi popravio pokrivanje komponenti — budite pažljivi kod promjene tih promptova jer to utječe na kvalitetu maski.
- HEIC / HEIF: notebook i backend podržavaju `pillow-heif` konverziju; u nekim okruženjima možda treba dodatna instalacija sustavskih paketa.

Treniranje YOLO modela (općenito)
1. Pripremi dataset u YOLO formatu (vidi `dataset.yaml` u ovoj mapi ako postoji).
2. Pokreni trening na Colab-u ili drugom GPU okruženju. Primjer koristeći `ultralytics`:
```bash
pip install ultralytics
# primjer (prilagodi parametre i putanje)
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```
3. Najbolji checkpoint će biti spremljen (`best.pt`); kopiraj ga u `ai/` ili Drive i koristite ga za inferencu u notebooku.

Kako se rezultati koriste u projektu
- `combined_detector.py` i drugi skripti u `neodata2/` čitaju `_masks.json` i izvode zonalnu analizu, računaju pokrivenost, i izvlače lista defekata. Te JSON-ove koristimo kao ulaz u evaluacijske i demo flow-ove.

Pokretanje lokalnih evaluacija
- Ako imaš sve rezultate u `segmented_output_improved/`, pokreni skripte iz `neodata2/` za analizu i spajanje rezultata (npr. `combined_detector.py` ili `final_report.py`).

Okruženje i ovisnosti
- Preporučeno: Python 3.9+
- Instaliraj iz `ai/requirements.txt` (ako postoji) ili ručno:
```bash
pip install pillow pillow-heif ultralytics opencv-python numpy matplotlib
```

Dodatne napomene za reproducibilnost
- Sačuvajte verzije paketa i preuzmite sve trained checkpoint-e na Drive — to značajno ubrzava ponovno izvođenje.
- Za arhiviranje rezultata držite strukturu `segmented_output_improved/{positive,negative}` i _masks.json objekte netaknutima.

Kontakt / dalje
- Ako želiš da dodam automatsko preuzimanje `segmented_output_improved/` iz Drive-a ili generator `dataset.yaml`, mogu to pripremiti kao skriptu.

---
Dokument napisao: tim Neodata — fokusirano na reproducibilnost i Colab pipeline za segmentaciju.
