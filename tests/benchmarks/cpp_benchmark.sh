#!/bin/bash

APP="./build/cpp/v2/noisy_cuda"
CSV="cpp_v2_benchmark.csv"

rm -f $CSV

RESOLUTIONS="128 983 1838 2694 3549 4404 5260 6115 6970 7826 8681 9536 10392 11247 12102 12958 13813 14668 15524 16384"

for RES in $RESOLUTIONS; do
    echo "------------------------------------------"
    echo "Esecuzione Risoluzione: ${RES}x${RES}"
    # Assicurati che APP sia definito (es. APP="./mio_programma")
    $APP --verbose --no-output --octaves 4 --size ${RES}x${RES}
done