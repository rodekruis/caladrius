conda env create -f caladriusenv.yml
conda clean -ay
cd caladrius/interface
npm install
cd client
npm install

