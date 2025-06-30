git clone https://github.com/biomed-AI/GPSite
mv GPSite/model/* ./Surf2Spot/data/model_gpsite
git clone https://github.com/JudeWells/Chainsaw
mv Chainsaw/saved_models/* ./Surf2Spot/data/chainsaw/saved_models
cd ./Surf2Spot/data/model_emb
git clone https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc
rm -rf GPSite
rm -rf Chainsaw
export PATH=$PATH:$(pwd)/Surf2Spot/data/feature_extraction/
chmod +x -R $(pwd)/Surf2Spot

