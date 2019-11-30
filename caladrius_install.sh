conda env create -f caladriusenv.yml
conda clean -ay
source activate caladriusenv
cd caladrius/interface
npm install
cd client
npm install

