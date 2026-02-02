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


cosa notiamo? memory bound

## Versione 2:

RICORDA DI RIMUOVERE FUNZIONI NON USATE

- Massimizzare l'Occupancy e Gestione Registri:
    - Scomponi la struct Vector2D in variabili scalari (float x, y) riducendo il register pressure
    - Aumento del Block Size (da 16 a 32 x 32 o 16)
    - total_noise da mantenere nei registri (velocissimi) per tutte le ottave e scritto in memoria globale una sola volta alla fine per eliminare il collo di bottiglia del bandwidth di memoria per i calcoli intermedi. (vedi v3)

- Ottimizzare l'ILP
    - spostare il ciclo delle ottave dentro il kernel
    - Loop Unrolling delle Ottave con #pragma unroll
    - Calcoli Indipendenti: Calcola i contributi dei quattro angoli (influence_top_left, ecc.) in modo che non dipendano l'uno dall'altro, permettendo l'esecuzione out-of-order delle istruzioni aritmetiche.

- Eliminazione di floorf e fmaxf e --use_fast_math


- riflettere su memorie (pinned, costant...)
    1. LUT: causa latenza perché viene consultata due volte per ogni ottava per trovare l'indice del gradiente.
        
        **Opzione A: Shared Memory (S-Mem)**, Caricamento collaborativo all'inizio del kernel.
        Perché: La Shared Memory ha latenze bassissime e gestisce meglio gli accessi casuali dei thread nello stesso warp.

        **Opzione B: Constant Memory (C-Mem)** Usare direttamente d_lookUpTable senza caricarla in S-Mem.
        Perché: La C-Mem ha una cache dedicata. È ottima se molti thread leggono lo stesso valore (broadcast), ma se i thread leggono indici molto diversi (come nel rumore), può generare conflitti serializzati.
    
    2. Gradients: Sono solo 8 vettori 2D.

        **Opzione A: Constant Memory (Standard)**  Idea: Lasciarli dove sono. 8 Vector2D occupano pochissimo spazio e rimangono quasi certamente nella cache L1/Constant.

        **Opzione B: Registri (Pre-caricamento)** Ogni thread carica gli 8 gradienti in variabili locali all'inizio.
        Perché: Accesso a latenza zero assoluta.

    3. Output / Accumulator

        **Opzione A: Fused Register Accumulation (Vincitore probabile)** Il ciclo delle ottave avviene dentro il kernel. Usi una variabile float total = 0 (registro). Scrivi in accumulator una sola volta alla fine.
        Perché: Riduce il traffico sulla memoria globale di un fattore pari al numero di ottave (N).

