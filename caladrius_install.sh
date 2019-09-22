conda env create -f caladriusenv.yml
conda clean -ay
cd caladrius/interface
yarn install
ln -s ../../../data src/data
