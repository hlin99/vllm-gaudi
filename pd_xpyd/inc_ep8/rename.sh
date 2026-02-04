for f in inc_*; do
    mv "$f" "${f#inc_}"
done

for f in *output*; do
    mv "$f" "${f//output/outputs}"
done

