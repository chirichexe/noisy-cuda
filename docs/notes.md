# CPP

## Versione 1

* **Descrizione generale**

- Implementazione naive della generazione di Perlin Noise 2D
- Strutture dati: 
    - `Vector2D`: vettore bidimensionale (2 float) con operazioni utili nell'algoritmo (sottrazione, prodotto scalare, lunghezza, normalizzazione), funzioni di fade e lerp
    - `Chunk`: rappresenta un blocco di rumore con dimensioni e coordinate, si occupa di generare i propri pixel dell'immagine complessiva

* **Considerazioni**

- La funzione rand() è gestita dal sistema operativo, non garantisce portabilità
- La matrice di gradienti è pesante in memoria, soluzioni online migliori (es. LUT)
- Perfezionamenti possibili sui tipi di dato scelti (es. size_t, uint8_t )
- Per implementazioni SIMD o CUDA potrebbe essere necessario reinterpretare le strutture dati realizzate (es. Chunk, Vector2D)
- Alleggerire operazioni sui vettori (es. evitare sqrt)

> Usare una LUT di gradienti significa sostituire la generazione casuale seguita da normalizzazione con un insieme finito di vettori già unitari e ben distribuiti. Questo elimina completamente la necessità di calcolare norme, divisioni e radici quadrate, che sono operazioni relativamente costose e difficili da ottimizzare in modo aggressivo dal compilatore. Il vantaggio non è solo aritmetico: il codice diventa più semplice e più prevedibile dal punto di vista delle prestazioni, perché il costo per gradiente si riduce a un accesso in memoria e a un prodotto scalare. Dal punto di vista numerico, una LUT consente di controllare esplicitamente la distribuzione angolare dei gradienti. Nel Perlin noise la qualità visiva dipende fortemente dall’isotropia del campo di gradienti; gradienti casuali normalizzati introducono inevitabilmente piccole anisotropie e bias statistici, mentre una LUT costruita su direzioni equispaziate sul cerchio garantisce simmetria e uniformità. Questo si traduce in un rumore più “pulito”, con meno artefatti direzionali e pattern ripetuti. C’è poi un aspetto di robustezza e portabilità. Con una LUT non dipendi dal comportamento di rand(), dalle differenze tra implementazioni, né da dettagli di floating point come underflow o precisione della normalizzazione. L’output diventa deterministico a parità di seed e dimensioni, e il codice non contiene operazioni delicate o potenzialmente problematiche dal punto di vista dello standard C++. Infine, l’uso di una LUT favorisce l’ottimizzazione a basso livello. Gli accessi ai gradienti sono regolari, facilmente prefetchabili e ben compatibili con la vectorization automatica. In un algoritmo come il tuo, dove il costo dominante è nel loop per pixel, questo tipo di regolarità è più prezioso di micro-ottimizzazioni aritmetiche isolate. In sintesi, la LUT sposta il problema da calcoli numerici ripetuti a una scelta strutturale migliore, con benefici simultanei su prestazioni, qualità e affidabilità.

> Le permutazioni servono a introdurre pseudo-casualità deterministica nella scelta dei gradienti partendo da una griglia regolare. Trasformano le coordinate intere in un indice apparentemente casuale, evitando pattern ripetitivi e allineamenti visibili nel noise. Allo stesso tempo garantiscono che lo stesso punto produca sempre lo stesso risultato, rendendo il rumore stabile e riproducibile. Sono usate perché sono semplici, veloci e permettono di controllare la periodicità del noise senza ricorrere a generatori casuali costosi.

# CUDA

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