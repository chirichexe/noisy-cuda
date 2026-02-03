# CPP

## Versione 1

* **Descrizione generale**

- Implementazione naive della generazione di Perlin Noise 2D
- Strutture dati: 
    - `Vector2D`: vettore bidimensionale (2 float) con operazioni utili nell'algoritmo (sottrazione, prodotto scalare, lunghezza, normalizzazione), funzioni di fade e lerp
    - `Chunk`: rappresenta un blocco di rumore con dimensioni e coordinate, si occupa di generare i propri pixel dell'immagine complessiva

- Viene generata una griglia fissa di vettori casuali (std::vector<Vector2D>) prima di iniziare il calcolo

* **Considerazioni**

- La funzione rand() è gestita dal sistema operativo, non garantisce portabilità
- La matrice di gradienti è pesante in memoria, soluzioni online migliori (es. LUT)
- Perfezionamenti possibili sui tipi di dato scelti (es. size_t, uint8_t )
- Per implementazioni SIMD o CUDA potrebbe essere necessario reinterpretare le strutture dati realizzate (es. Chunk, Vector2D)
- Alleggerire operazioni sui vettori (es. evitare sqrt)

> Usare una LUT di gradienti significa sostituire la generazione casuale seguita da normalizzazione con un insieme finito di vettori già unitari e ben distribuiti. Questo elimina completamente la necessità di calcolare norme, divisioni e radici quadrate, che sono operazioni relativamente costose e difficili da ottimizzare in modo aggressivo dal compilatore. Il vantaggio non è solo aritmetico: il codice diventa più semplice e più prevedibile dal punto di vista delle prestazioni, perché il costo per gradiente si riduce a un accesso in memoria e a un prodotto scalare. Dal punto di vista numerico, una LUT consente di controllare esplicitamente la distribuzione angolare dei gradienti. Nel Perlin noise la qualità visiva dipende fortemente dall’isotropia del campo di gradienti; gradienti casuali normalizzati introducono inevitabilmente piccole anisotropie e bias statistici, mentre una LUT costruita su direzioni equispaziate sul cerchio garantisce simmetria e uniformità. Questo si traduce in un rumore più “pulito”, con meno artefatti direzionali e pattern ripetuti. C’è poi un aspetto di robustezza e portabilità. Con una LUT non dipendi dal comportamento di rand(), dalle differenze tra implementazioni, né da dettagli di floating point come underflow o precisione della normalizzazione. L’output diventa deterministico a parità di seed e dimensioni, e il codice non contiene operazioni delicate o potenzialmente problematiche dal punto di vista dello standard C++. Infine, l’uso di una LUT favorisce l’ottimizzazione a basso livello. Gli accessi ai gradienti sono regolari, facilmente prefetchabili e ben compatibili con la vectorization automatica. In un algoritmo come il tuo, dove il costo dominante è nel loop per pixel, questo tipo di regolarità è più prezioso di micro-ottimizzazioni aritmetiche isolate. In sintesi, la LUT sposta il problema da calcoli numerici ripetuti a una scelta strutturale migliore, con benefici simultanei su prestazioni, qualità e affidabilità.

## Versione 2
- Uso di Lookup Table di 512 interi e array statico di 8 gradienti fissi. Per due motivi:

    1. Coerenza: Lo shuffle iniziale (Fisher-Yates) basato sul seed garantisce che lo stesso seed generi sempre la stessa LUT, e quindi lo stesso identico mondo. la tabella è un riferimento fisso, siamo sicuri che le coordinate (10, 20) daranno sempre gli stessi valori

    2. Velocità: Dare ai core una tabella di numeri pronti da consultare permette loro di calcolare milioni di pixel al secondo.

- Efficienza calcoli:

    - L'operatore & 255: Questo è il "modulo veloce". Invece di usare % (lento), usiamo un'operazione bitwise per restare nel range [0,255].

    - Double Lookup: lookUpTable[lookUpTable[xi] + yi] è un trucco matematico per mescolare le coordinate e ottenere un indice pseudo-casuale unico per ogni vertice della griglia.

    - & 7: Poiché hai definito 8 gradienti, estraiamo solo gli ultimi 3 bit per scegliere quale vettore usare.



> Le permutazioni servono a introdurre pseudo-casualità deterministica nella scelta dei gradienti partendo da una griglia regolare. Trasformano le coordinate intere in un indice apparentemente casuale, evitando pattern ripetitivi e allineamenti visibili nel noise. Allo stesso tempo garantiscono che lo stesso punto produca sempre lo stesso risultato, rendendo il rumore stabile e riproducibile. Sono usate perché sono semplici, veloci e permettono di controllare la periodicità del noise senza ricorrere a generatori casuali costosi.

# CUDA

Test eseguiti su:

NVIDIA GeForce RTX 3050 6GB

configurazioni test:

Configuration:
  Size:        2048 x 2048
  Format:      png
  No output:   enabled
  Offset:      (0, 0)
  Seed:        1545174314 (auto-generated)
  Verbose:     enabled

Generating Perlin noise with options:
  freq=50
  amp=1
  octaves=10
  lacunarity=2
  persistence=0.5

## Versione 1

* **Descrizione generale**

- Implementazione di Perlin Noise in CUDA (uguale alla versione 2 C++). 
- Strutture dati: 
    - `Vector2D`: vettore bidimensionale (2 float) con operazioni utili nell'algoritmo, funzioni di fade e lerp con i tag specifici per essere eseguiti sul device

- Rimossa la struttura `Chunk`, l'algoritmo viene eseguito direttamente in un SM del device

- La `CHUNK_SIZE_LENGHT` è diventata `BLOCK_SIZE` impostata a 16. Perchè?
    - Le GPU NVIDIA eseguono i thread in gruppi di 32, chiamati Warp. Un blocco da 16×16=256 thread contiene esattamente 8 warp (256/32=8).
    - Usa molti registri (Vector2D, calcoli matematici...), un blocco da 1024 thread (il massimo) potrebbe impedire ad altri blocchi di caricarsi, lasciando l'SM sotto-utilizzato.
  
  16x16 è la via di mezzo perfetta: è abbastanza grande da saturare i warp, ma abbastanza piccolo da permettere a più blocchi di risiedere contemporaneamente sullo stesso SM, aiutando latency hiding.
    
    (mettere screen occupancy)

- Copia elementi:
    - `gradients` e `lookUpTable`: memoria costante (R only)
    - `accumulator`: memoria globale (R - W)


cosa notiamo?
- Occupancy ok
- lanciamo un kernel pper ogni ottava
- siamo memory bound (mettiamo tutto in memoria globale)

## Versione 2:

RICORDA DI RIMUOVERE FUNZIONI NON USATE

Dalle slide del pdf memoria comune, il nsotro kernel è "naturalmente" compute bound ma facendo benchmark da nsight compute della v1 si vede che stiamo sfruttando male la memoria (siamo memory bound).

L’obiettivo è aumentare l’arithmetic intensity e spostare il punto del kernel verso l’alto-destra nel Roofline 

**Metriche da controllare (in generale)**
- Arithmetic Intensity (FLOP/byte) Per vedere se il kernel si sta spostando verso destra nel Roofline.
- Guardare roofline model

### Ottimizzaioni possibili per migliorare la computazione e occupancy:

**Massimizzare l'Occupancy e Gestione Registri**
- Scomporre la struct Vector2D in variabili scalari (float x, y) riducendo il register pressure (da vedere con filippo, non so quanto convenga davvero)
- Aumento del Block Size (da 16 a 32 x 32 o 16) (da testare sperimentalmente, di sicuro sopra i 16, non dobbiamo essere per forza rettangolari)
- total_noise (noise totale calcolato da ogni ottava eseguita da un thread) da mantenere nei registri (che sono per definizione veloci) per tutte le ottave e scritto in memoria globale **una sola volta** alla fine per eliminare il collo di bottiglia del bandwidth di memoria per i calcoli intermedi. (vedi v3)

    **Metriche da controllare**
    - Registers per thread: alti == meno occupancy, meno latency hiding
    - Active warps/SM: serve a tenere occupati gli SM mentre si nasconde la latenza.
    - Register spilling: spill -> accessi DRAM -> kernel diventa memory bound


**Ottimizzare l'ILP**
- spostare il ciclo delle ottave dentro il kernel
- Loop Unrolling delle Ottave con `#pragma unroll` 
- Calcola i contributi dei quattro angoli (influence_top_left, ecc.) in modo che non dipendano l'uno dall'altro, permettendo l'esecuzione out-of-order delle istruzioni aritmetiche. (da vedere con filippo come e se farlo)
- Eliminazione di floorf e fmax (PESANTI) e --use_fast_math (da vedere)

    **Metriche da controllare**
    - Instructions per cycle (IPC): misura quanto bene lo scheduler riesce a sovrapporre istruzioni.
    - Eligible warps per cycle: indica se l’SM ha abbastanza lavoro indipendente da eseguire.
    - Stall reason: execution dependency. se è alto == poco ILP, dipendenze seriali (es. accumulatore scritto troppo spesso).

### Ottimizzaioni possibili per migliorare USO DI MEMORIA

Qui ho messo cose da valutare sperimentalmente insieme:

-  LUT: causa latenza perché viene consultata due volte per ogni ottava per trovare l'indice del gradiente.
        
    **Opzione A: Shared Memory (S-Mem)**, Caricamento collaborativo all'inizio del kernel.
    Perché: La Shared Memory ha latenze bassissime e gestisce meglio gli accessi casuali dei thread nello stesso warp.

    **Opzione B: Constant Memory (C-Mem)** Usare direttamente d_lookUpTable senza caricarla in S-Mem.
    Perché: La C-Mem ha una cache dedicata. È ottima se molti thread leggono lo stesso valore (broadcast), ma se i thread leggono indici molto diversi (come nel rumore), può generare conflitti serializzati.
    
- Gradients: Sono solo 8 vettori 2D.

    **Opzione A: Constant Memory (Standard)**  Idea: Lasciarli dove sono. 8 Vector2D occupano pochissimo spazio e rimangono quasi certamente nella cache L1/Constant.

    **Opzione B: Registri (Pre-caricamento)** Ogni thread carica gli 8 gradienti in variabili locali all'inizio.
        Perché: Accesso a latenza zero assoluta.

- Output / Accumulator

    **Opzione A: Fused Register Accumulation (Vincitore probabile)** Il ciclo delle ottave avviene dentro il kernel. Usi una variabile float total = 0 (registro). Scrivi in accumulator una sola volta alla fine.
    Perché: Riduce il traffico sulla memoria globale di un fattore pari al numero di ottave (N).

**Metriche da controllare**

1. Accessi a memoria globale
    - Global load/store transactions: da ridurli (es. accumulo in registro, scrittura finale unica).
    - Bytes read/written per thread o per pixel: per verificare l’aumento di arithmetic intensity.
    - Memory bandwidth (GB/s)

2. Cache
    - L1 cache hit / miss rate. miss elevati == pattern irregolare.
    - L2 cache hit / miss rate. miss elevati == il working set è troppo grande o troppo disperso.

3. Constant vs Shared
- Constant memory hit: basso == accessi non broadcast, memoria **costante** non adatta.
- Shared memory load efficiency / bank conflicts: conflitti annullano il vantaggio della **Shared-Mem**.

---

Idealmente, confifugrazione migliore:
- accumulo per ottava: registri -> buffer finale: Global Memory
- gradienti(8) -> memoria costante / registri
- LookUpTable -> shared memory 
caricare solo sottoinsiemi locali, snippet di codice:
- bonus finale: memoria pinned (da guardare)

```
__global__ void perlin_kernel(...)
{
    // Shared LUT (512 int)
    __shared__ int s_perm[512];

    // Linear thread index in the block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_threads = blockDim.x * blockDim.y;

    // Cooperative load: each thread loads multiple elements
    for (int i = tid; i < 512; i += block_threads) {
        s_perm[i] = d_lookUpTable[i];
    }

    // Synchronize to ensure LUT is fully loaded
    __syncthreads();

    // ---- from here on, use s_perm instead of d_lookUpTable ----
}
```

# SIMD

considerazione sui tipi di dato