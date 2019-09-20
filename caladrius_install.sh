conda env create -f caladriusenv.yml
conda activate caladriusenv
cd caladrius/interface
npm install
ln -s ../../../data src/data

