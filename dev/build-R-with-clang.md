curl -O https://cran.r-project.org/src/base/R-4/R-4.6.0.tar.gz
tar -xf R-4.6.0.tar.gz
cd R-4.6.0

export CC=/usr/bin/clang
export CXX=/usr/bin/clang++

./configure \
  --prefix=/opt/R-4.6.0 \
  --enable-R-shlib \
  --with-blas="-lopenblas" \
  --with-lapack="-lopenblas" \
  --with-x \
  --with-readline \
  CFLAGS="-O3 -Wall -fno-lto" \
  CXXFLAGS="-O3 -Wall -fno-lto" \
  CXX11FLAGS="-O3 -Wall -fno-lto" \
  CXX14FLAGS="-O3 -Wall -fno-lto" \
  CXX17FLAGS="-O3 -Wall -fno-lto" \
  CXX20FLAGS="-O3 -Wall -fno-lto"

make -j$(nproc)

sudo make install

sudo ln -sf /opt/R-4.6.0/bin/R /usr/local/bin/R
sudo ln -sf /opt/R-4.6.0/bin/Rscript /usr/local/bin/Rscript
