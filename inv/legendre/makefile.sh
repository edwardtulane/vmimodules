#! /bin/bash


gcc  -fPIC -O2 -Wall -c legendre_polynomial.c
gcc  -fPIC -O2 -Wall -c abel.c
gcc  -shared -fPIC -O2 -Wall legendre_polynomial.o abel.o -o abel.so
