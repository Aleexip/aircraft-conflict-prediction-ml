# aircraft-conflict-prediction-ml

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Panait Ionut-Alexandru
**Data:** 21/11/2025

1. project-name/
   ├── README.md
   ├── docs/
   │ └── datasets/ # Diagrame distributie viteze, exemple de traiectorii conflictuale
   ├── data/
   │ ├── raw/ # Date simulate brute (JSON/CSV) - traiectorii generate
   │ ├── processed/ # Date normalizate și transformate în secvențe (Tensori)
   │ ├── train/ # Set de instruire (secvente input + label)
   │ ├── validation/ # Set de validare
   │ └── test/ # Set de testare
   ├── src/
   │ ├── preprocessing/ # Scripturi: coord_transform.py, normalization.py, sliding_window.py
   │ ├── data_acquisition/ # Script: trajectory_generator.py (Simulator Python)
   │ └── neural_network/ # (Urmează în etapa 4)
   ├── config/ # Configuraii simulare (limite viteză, rată eșantionare)
   └── requirements.txt # numpy, pandas, scikit-learn, matplotlib

2. Descrierea Setului de Date

2.1. Sursa datelor

Origine: Date generate pe baza ecuatiilor de miscare cinematica a aeronavelor, inspirate din date reale OpenSky Network
Modul de achizitie: Generare din script (Python)

Perioada / conditiile colecatarii: Simulare scenarii de zbor de croaziera (En-route) la altitudini de FL300-FL400.

2.2. Caracteristicile dataset-ului

Numar total de observatii: 10,000 de perechi de traiectorii (scenarii a cate 2 minute fiecare).

Numar de caracteristici (features): 14 (câte 7 per aeronavă) per pas de timp.

Tipuri de date: Numerice (Float64) / Temporale (Serii de timp).

Format fisiere: CSV (pentru raw) / NPY (NumPy binary pentru datele procesate).

2.3. Descrierea fiecarei caracteristici (Per pas de timp t)

Caracteristica Tip Unitate Descriere Domeniu valori
relative_x numeric m Poziția X relativă -50km ... +50km
relative_y numeric m Poziția Y relativă fata de punctul de referinta -50km ... +50km
altitudine numeric ft Altitudinea barometrică 30,000 – 40,000
v_ground numeric m/s Viteza la sol (Ground Speed) 200 – 300 m/s
heading numeric rad Directia de deplasare (Heading) -pi ... +pi
v_vertical numeric m/s Rata de urcare/coborare -15 ... +15
bank_angle numeric rad Unghiul de inclinare (pentru viraje) -0.5 ... +0.5
label_conflict binar - Variabila Tinta (Target): 1 = Conflict, 0 = Safe {0, 1}

3. Analiza Exploratorie a Datelor (EDA) – Sintetic
   3.1 Statistici descriptive aplicate
   Distributia Distantelor Minime: Histograme pentru a verifica cati dintre scenarii ajung sub pragul de siguranta (ex: 5NM orizontal).

Analiza Cinematica: Verificarea realismului vitezelor și acceleratiilor (ex: nicio aeronava nu vireaza instantaneu cu 90 de grade).

Corelatii: Matricea de corelatie între v_vertical și schimbarea altitudinii.

3.2 Analiza calității datelor
Zgomot: Datele brute conțin zgomot Gaussian adăugat intenționat (mu=0, sigma=15m) pentru a simula eroarea GPS/Radar.

Consistenta Temporala: Verificarea pasului de timp (Delta t = 1s) – nu există "sarituri" temporale.

3.3 Probleme identificate și soluții
Dezechilibru de clasa: În aviatia reala, conflictele sunt extrem de rare (<0.01%).
Solutie: Generatorul de date a fost configurat sa produca un dataset echilibrat artificial: 50% scenarii de conflict, 50% scenarii sigure (Safe), pentru ca reteaua sa poată invata caracteristicile ambelor clase.

4. Preprocesarea Datelor

4.1 Curatarea datelor
Conversie Coordonate: Transformarea din coordonate sferice (Lat/Lon) în coordonate Carteziene Locale (X, Y) pentru a facilita calculele euclidiene ale rețelei.

Eliminarea scenariilor nerealiste: Filtrarea traiectoriilor care implică forțe G mai mari de 2G (manevre imposibile pentru avioane comerciale).

4.2 Transformarea caracteristicilor

Normalizare (Scalare):
Deoarece altitudinea (30,000 ft) și viteza verticala (10 m/s) au ordine de marime diferite, se aplică Min-Max Scaling pentru a aduce toate valorile în intervalul [0, 1] sau StandardScaler (media 0, deviația 1).

Sliding Window (Fereastră Glisanta):
Transformarea datelor tabulare în secvențe 3D pentru LSTM.
Dimensiune fereastră: 30 pași de timp (ultimele 30 secunde).

4.3 Structurarea seturilor de date
Impartire:70% – Train: Pentru antrenarea ponderilor.
15% – Validation: Pentru reglarea parametrilor și Early Stopping.
15% – Test: Pentru evaluarea finala si compararea cu Baseline-ul geometric.

Principii respectate:
Shuffle la nivel de scenariu: Nu amestecam punctele individuale de timp (pentru a pastra secventialitatea), ci amestecam ordinea scenariilor de zbor complete inainte de impartire.

4.4 Salvarea rezultatelor preprocesarii

Datele sunt salvate ca tensori .npy (NumPy) pentru încarcare rapida în PyTorch/TensorFlow.

5. Fisiere Generate in Aceasta Etapa

- `data/raw/` – date brute
- `data/processed/` – date curățate & transformate
- `data/train/`, `data/validation/`, `data/test/` – seturi finale
- `src/preprocessing/` – codul de preprocesare
- `data/README.md` – descrierea dataset-ului

6. Stare Etapă (de completat de student)

- [x ] Structură repository configurată
- [] Dataset analizat (EDA realizată)
- [] Date preprocesate
- [ ] Seturi train/val/test generate
- [x ] Documentație actualizată în README + `data/README.md`
