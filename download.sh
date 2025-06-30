git clone https://github.com/biomed-AI/GPSite
mv GPSite/model/* ./surf2spot/data/model_gpsite
git clone https://github.com/JudeWells/Chainsaw
mv Chainsaw/saved_models/* ./surf2spot/data/chainsaw/saved_models
cd ./surf2spot/data/model_emb
git clone https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc
rm -rf GPSite
rm -rf Chainsaw
export PATH=$PATH:$(pwd)/surf2spot/data/feature_extraction/
