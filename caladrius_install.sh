conda env create -f caladriusenv.yml
cd caladrius/interface
yarn install
ln -s ../../../data src/data
