# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Panait Ionut-Alexandru 
**Link Repository GitHub** https://github.com/Aleexip/aircraft-conflict-prediction-ml
**Data:** 05/12/2025 
---

## Scopul Etapei 4

Această etapă corespunde punctului **5. Dezvoltarea arhitecturii aplicației software bazată pe RN** din lista de 9 etape - slide 2 **RN Specificatii proiect.pdf**.

**Trebuie să livrați un SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA). In acest stadiu modelul RN este doar definit și compilat (fără antrenare serioasă).**

### IMPORTANT - Ce înseamnă "schelet funcțional":

 **CE TREBUIE SĂ FUNCȚIONEZE:**
- Toate modulele pornesc fără erori
- Pipeline-ul complet rulează end-to-end (de la date → până la output UI)
- Modelul RN este definit și compilat (arhitectura există)
- Web Service/UI primește input și returnează output

 **CE NU E NECESAR ÎN ETAPA 4:**
- Model RN antrenat cu performanță bună
- Hiperparametri optimizați
- Acuratețe mare pe test set
- Web Service/UI cu funcționalități avansate

**Scopul anti-plagiat:** Nu puteți copia un notebook + model pre-antrenat de pe internet, pentru că modelul vostru este NEANTRENAT în această etapă. Demonstrați că înțelegeți arhitectura și că ați construit sistemul de la zero.

---

##  Livrabile Obligatorii

### 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software (max ½ pagină)
Completați in acest readme tabelul următor cu **minimum 2-3 rânduri** care leagă nevoia identificată în Etapa 1-2 cu modulele software pe care le construiți (metrici măsurabile obligatoriu):

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Alertarea din timp a controlorilor (ATCO) in scenarii de viraj | Predictie probabilistica a coliziunii bazata pe istoricul traiectoriei (LSTM) (Long Short-Term Memory) in loc de vectori liniari | src/neural_network/ + src/app/|
| Reducerea alarmelor false în zone aglomerate | Clasificare binară (Conflict/Safe) cu o fereastră de analiză de 30 secunde | src/preprocessing/ + src/neural_network/ |
| Simulare rapidă a scenariilor de risc pentru antrenament | Generarea automata a 2000+ scenarii de zbor sintetice cu zgomot realist | src/data_acquisition/ |


**Instrucțiuni:**
- Fiți concreti (nu vagi): "detectare fisuri sudură" ✓, "îmbunătățire proces" ✗
- Specificați metrici măsurabile: "< 2 secunde", "> 95% acuratețe", "reducere 20%"
- Legați fiecare nevoie de modulele software pe care le dezvoltați

---

### 2. Contribuția Voastră Originală la Setul de Date – MINIM 40% din Totalul Observațiilor Finale

**Regula generală:** Din totalul de **N observații finale** în `data/processed/`, **minimum 40%** trebuie să fie **contribuția voastră originală**.

||| Total observatii finale: 2000 scenarii (cca. 240,000 randuri/frame-uri) Observatii originale: 2000 (100%), toate sunt generate de script.


#### Tipuri de contribuții acceptate (exemple din inginerie):



#### Declarație obligatorie în README:

Scrieți clar în acest README (Secțiunea 2):

```markdown
### Contribuția originală la setul de date:

**Total observații finale:** [N] (după Etapa 3 + Etapa 4)
**Observații originale:** [M] ([X]%)

**Tipul contribuției:**
[X] Date generate prin simulare fizică  
[ ] Date achiziționate cu senzori proprii  
[ ] Etichetare/adnotare manuală  
[ ] Date sintetice prin metode avansate  

**Descriere detaliată:**
[Explicați în 2-3 paragrafe cum ați generat datele, ce metode ați folosit, 
de ce sunt relevante pentru problema voastră, cu ce parametri ați rulat simularea/achiziția]

Descriere detaliata: Am dezvoltat un motor de simulare fizică în Python (trajectory_generator.py) care modeleaza ecuatiile cinematice de miscare pentru două aeronave. Scriptul generează traiectorii realiste incluzand:
  -Viteze variabile (200-250 m/s).
  -Manevre de viraj (rate de girație variabile) pentru a depăși limitările algoritmilor liniari clasici.
  -Injecție de zgomot Gaussian (mu=0, sigma=100m$) pentru a simula erorile senzorilor radar/GPS reali.
  -Etichetare automata bazata pe calculul distantei euclidiene minime (prag 5 NM).

**Locația codului:** `Locatia codului: src/data_acquisition/trajectory_generator.py`
**Locația datelor:** `Locația datelor: data/raw/simulated_trajectories.csv`

**Dovezi:**
Grafic traiectorii generate: docs/datasets/trajectory_example.png
Statistici distributie clase (Safe vs Conflict): Vizibile în log-ul de generare.
```

#### Exemple pentru "contribuție originală":
-Simulări fizice realiste cu ecuații și parametri justificați  
-Date reale achiziționate cu senzori proprii (setup documentat)  
-Augmentări avansate cu justificare fizică (ex: simulare perspective camera industrială)  


#### Atenție - Ce NU este considerat "contribuție originală":

- Augmentări simple (rotații, flips, crop) pe date publice  
- Aplicare filtre standard (Gaussian blur, contrast) pe imagini publice  
- Normalizare/standardizare (aceasta e preprocesare, nu generare)  
- Subset dintr-un dataset public (ex: selectat 40% din ImageNet)


---

### 3. Diagrama State Machine a Întregului Sistem (OBLIGATORIE)

Justificarea State Machine-ului ales: Am ales o arhitectura de tip Batch Processing & Simulation Pipeline. Deoarece sistemul este unul de asistentă critica (Safety Critical), fluxul este secvential și strict controlat: incepe cu generarea/incarcarea scenariului, trece prin validare și preprocesare (transformare în tensori), urmata de inferenta modelului LSTM si afisarea alertei. Starea de eroare (ERROR) trateaza cazurile de fisiere corupte sau traiectorii incomplete.

Descrierea Starilor:

IDLE: Sistemul asteapta input de la utilizator (Incarcare CSV sau generare scenariu nou).

LOAD_SCENARIO: Citirea datelor brute și validarea formatului coloanelor.

PREPROCESS: Normalizarea datelor (MinMax) si crearea ferestrelor glisante (Sliding Window 30 sec).

INFERENCE: Modelul LSTM proceseaza tensorul (Batch, 30, 14) si returneaza probabilitatea de coliziune.

DISPLAY_RESULT: Afisarea grafica a traiectoriilor si a verdictului (SAFE/CONFLICT).

**Cerințe:**
- **Minimum 4-6 stări clare** cu tranziții între ele
- **Formate acceptate:** PNG/SVG, pptx, draw.io 
- **Locație:** `docs/state_machine.*` (orice extensie)
- **Legendă obligatorie:** 1-2 paragrafe în acest README: "De ce ați ales acest State Machine pentru nevoia voastră?"

**Stări tipice pentru un SIA:**
```
IDLE → ACQUIRE_DATA → PREPROCESS → INFERENCE → DISPLAY/ACT → LOG → [ERROR] → STOP
                ↑______________________________________________|
```
IDLE → USER_ACTION (Generate/Upload) → LOAD_SCENARIO 
       ↓
    VALIDATE_DATA 
       ├─ [Invalid] → ERROR → LOG_ERROR → IDLE
       └─ [Valid] → PREPROCESS (Normalize & Windowing)
                        ↓
                    RN_INFERENCE (LSTM Model)
                        ↓
                    DECISION_LOGIC (Threshold > 0.5)
                        ↓
                    DISPLAY_RESULT (Plot & Alert) → IDLE




**Notă pentru proiecte simple:**
Chiar dacă aplicația voastră este o clasificare simplă (user upload → classify → display), trebuie să modelați fluxul ca un State Machine. Acest exercițiu vă învață să gândiți modular și să anticipați toate stările posibile (inclusiv erori).

**Legendă obligatorie (scrieți în README):**
```markdown
### Justificarea State Machine-ului ales:

Am ales arhitectura [descrieți tipul: monitorizare continuă / clasificare la senzor / 
predicție batch / control în timp real] pentru că proiectul nostru [explicați nevoia concretă 
din tabelul Secțiunea 1].

Stările principale sunt:
1. [STARE_1]: [ce se întâmplă aici - ex: "achiziție 1000 samples/sec de la accelerometru"]
2. [STARE_2]: [ce se întâmplă aici - ex: "calcul FFT și extragere 50 features frecvență"]
3. [STARE_3]: [ce se întâmplă aici - ex: "inferență RN cu latență < 50ms"]
...

Tranzițiile critice sunt:
- [STARE_A] → [STARE_B]: [când se întâmplă - ex: "când buffer-ul atinge 1024 samples"]
- [STARE_X] → [ERROR]: [condiții - ex: "când senzorul nu răspunde > 100ms"]

Starea ERROR este esențială pentru că [explicați ce erori pot apărea în contextul 
aplicației voastre industriale - ex: "senzorul se poate deconecta în mediul industrial 
cu vibrații și temperatură variabilă, trebuie să gestionăm reconnect automat"].

Bucla de feedback [dacă există] funcționează astfel: [ex: "rezultatul inferenței 
actualizează parametrii controlerului PID pentru reglarea vitezei motorului"].
```

---

### 4. Scheletul Complet al celor 3 Module Cerute la Curs (slide 7)

Toate cele 3 module trebuie să **pornească și să ruleze fără erori** la predare. Nu trebuie să fie perfecte, dar trebuie să demonstreze că înțelegeți arhitectura.

| **Modul** | **Python (exemple tehnologii)** | **LabVIEW** | **Cerință minimă funcțională (la predare)** |
|-----------|----------------------------------|-------------|----------------------------------------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/` | LLB cu VI-uri de generare/achiziție | **MUST:** Produce CSV cu datele voastre (inclusiv cele 40% originale). Cod rulează fără erori și generează minimum 100 samples demonstrative. |
| **2. Neural Network Module** | `src/neural_network/model.py` sau folder dedicat | LLB cu VI-uri RN | **MUST:** Modelul RN definit, compilat, poate fi încărcat. **NOT required:** Model antrenat cu performanță bună (poate avea weights random/inițializați). |
| **3. Web Service / UI** | Streamlit, Gradio, FastAPI, Flask, Dash | WebVI sau Web Publishing Tool | **MUST:** Primește input de la user și afișează un output. **NOT required:** UI frumos, funcționalități avansate. |

Modul,Tehnologie,Status Implementare
1. Data Acquisition,"Python (numpy, pandas)",Functional. Scriptul trajectory_generator.py genereaza datele brute.
2. Neural Network,Python (TensorFlow/Keras),"Functional. Clasa ConflictModel este definita, modelul LSTM se compileaza."
3. Web Service / UI,Python (Streamlit),Functional. Interfata permite vizualizarea datelor și rularea inferenței.

#### Detalii per modul:

Modul 1: Data Acquisition (src/data_acquisition/)
  -Ruleaza fara erori și produce fisierul .csv cu 14 coloane.
  -Include logica de fizica pentru miscare si detectare coliziuni.

Modul 2: Neural Network (src/neural_network/)
Fisier: model_def.py.
Arhitectura: LSTM (Input: 30 steps) -> LSTM Layer -> Dense -> Output (Sigmoid).
Modelul este instantiat si compilat cu BinaryCrossentropy, gata de antrenare.

Modul 3: Web Service / UI (src/app/)
Fisier: app.py.
Interfata Web simpla (Streamlit) care:
Buton "Generează Scenariu Nou".
Afiseaza graficul traiectoriilor (Matplotlib).
Apeleaza modelul (neantrenat momentan) și afisează o probabilitate initiala.

**Funcționalități obligatorii:**

#### **Modul 2: Neural Network Module**

**Funcționalități obligatorii:**
- [ ] Arhitectură RN definită și compilată fără erori
- [ ] Model poate fi salvat și reîncărcat
- [ ] Include justificare pentru arhitectura aleasă (în docstring sau README)
- [ ] **NU trebuie antrenat** cu performanță bună (weights pot fi random)


#### **Modul 3: Web Service / UI**



## Checklist Final – Bifați Totul Înainte de Predare

[x] Tabelul Nevoie → Solutie completat.
[x] Declaratie contributie 100% date originale (Simulare).
[x] Diagrama State Machine explicata și justificata.
[x] Modul 1: Script generare date funcțional.
[x] Modul 2: Arhitectură LSTM definită în cod.
[x] Modul 3: UI Streamlit funcțional (afiseaza datele).
[x] Structura de foldere respectată.

**Predarea se face prin commit pe GitHub cu mesajul:**  
`"Etapa 4 completă - Arhitectură SIA funcțională"`

**Tag obligatoriu:**  
`git tag -a v0.4-architecture -m "Etapa 4 - Skeleton complet SIA"`


